"""Per-chunk timing and memory for WorldKV's generation loop.

WorldKV calls this from wan/image2video_fast.py with four lines:

    telemetry = ChunkTelemetry.from_env()   # before the chunk loop
    telemetry.start_chunk()                 # first line of the loop body
    telemetry.end_chunk(chunk_size)         # last line of the loop body
    telemetry.save()                        # after the loop (on every rank)

It records nothing unless the WORLDKV_TELEMETRY environment variable names
an output JSON file (eval.runner sets it next to each video). Only torch and
the standard library are used, so it runs in WorldKV's own environment.

What is measured:
  - time per chunk, from the start of the chunk (memory retrieval, including
    any CPU -> GPU copy) to the end (denoising, clean KV pass, storing the
    chunk). Model loading and VAE decoding are not included.
  - FPS per chunk = video frames in the chunk / chunk time. A chunk of 3
    latent frames is 12 video frames, except the first chunk (9 frames).
  - GPU memory per GPU: allocated and reserved, per chunk and at the peak.
  - CPU memory (RSS) per chunk, of rank 0 and summed over all ranks (one
    process per GPU), and the true peak of each process (Linux VmHWM, reset
    when the run starts, so earlier runs of the same worker do not count).
  - Retrieval bank: bytes held by the KV bank, summed over all ranks, right
    after each chunk is stored (its largest point in the chunk), and the
    chunk ids retrieved for each chunk. Set by two extra hooks in the loop:

    telemetry.record_retrieval(selected_ids)
    telemetry.record_bank(bank_bytes(kv_bank_k, kv_bank_v))

If a run crashes (e.g. out of GPU memory), save() never runs. So each rank
also rewrites a small progress file after every chunk (`<path>.rank<r>.progress.json`,
removed by save()); eval.runner reads it to report how far the run got.

GPUs run asynchronously, so on a GPU the chunk times come from CUDA events
recorded on the generation stream: a start and an end marker per chunk, which
the GPU timestamps when it reaches them. Recording a marker does not wait for
anything, so work on other streams (e.g. background CPU <-> GPU copies) keeps
running across chunk boundaries. The times are read in save(), after the
last chunk, waiting only for the generation stream. Pass `stream` when the
generation does not run on the current stream. Without a GPU, the CPU clock
is used.
"""

import json
import os
import time

import torch
import torch.distributed as dist

ENV_VAR = "WORLDKV_TELEMETRY"
VAE_STRIDE_T = 4
GB = 1024 ** 3


def _proc_status_gb(field):
    """A memory field of /proc/self/status (e.g. VmRSS, VmHWM) in GB, or None."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return None


def _cpu_rss_gb():
    """Current resident memory of this process in GB, or None if unknown."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / GB
    except ImportError:
        return _proc_status_gb("VmRSS")


def _reset_cpu_peak():
    """Restart VmHWM (peak RSS) from the current RSS; False if not possible."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _gather(obj):
    """`obj` from every rank, as a list (one item when not distributed)."""
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, obj)
        return gathered
    return [obj]


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _sum_or_none(values):
    return None if None in values else sum(values)


def _sum_per_chunk(per_rank):
    """Per-chunk totals over ranks (None for a chunk where any rank has no value)."""
    return [_sum_or_none(list(values)) for values in zip(*per_rank)]


class ChunkTelemetry:

    def __init__(self, path=None, stream=None):
        self.path = path
        self.enabled = path is not None
        self.cuda = torch.cuda.is_available()
        self.stream = stream  # None: the current stream when each marker is recorded
        self.chunk_times_s = []
        self.chunk_video_frames = []
        self.gpu_mem_allocated_gb = []
        self.gpu_mem_reserved_gb = []
        self.cpu_rss_gb = []
        self.bank_gb = []  # this rank's bank after each chunk is stored (None: no bank)
        self.retrieved_chunks = []
        self._retrieval = []
        self._bank = None
        self._start = None
        self._events = []  # (start, end) CUDA events per chunk, read in save()
        self.cpu_peak_reset = False
        self.cpu_rss_start_gb = None
        if self.enabled:
            # Peaks count from here: model weights already loaded are included.
            if self.cuda:
                torch.cuda.reset_peak_memory_stats()
            self.cpu_peak_reset = _reset_cpu_peak()
            self.cpu_rss_start_gb = _cpu_rss_gb()
            self.progress_path = f"{path}.rank{_rank()}.progress.json"

    @classmethod
    def from_env(cls, stream=None):
        return cls(os.environ.get(ENV_VAR) or None, stream)

    def _mark(self):
        """A marker on the generation stream (GPU), or the CPU clock."""
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream if self.stream is not None else torch.cuda.current_stream())
        return event

    def start_chunk(self):
        if not self.enabled:
            return
        self._start = self._mark()
        self._retrieval = []
        self._bank = None

    def record_retrieval(self, chunk_ids):
        """The stored chunks retrieved for the current chunk."""
        if self.enabled:
            self._retrieval = list(chunk_ids)

    def record_bank(self, nbytes):
        """Bytes held by this rank's retrieval bank right after the chunk is stored."""
        if self.enabled:
            self._bank = nbytes / GB

    def end_chunk(self, latent_frames):
        """Close the current chunk, which generated `latent_frames` latent frames."""
        if not self.enabled:
            return
        end = self._mark()
        if self.cuda:
            self._events.append((self._start, end))
        else:
            self.chunk_times_s.append(end - self._start)
        frames = VAE_STRIDE_T * latent_frames
        if not self.chunk_video_frames:
            frames -= VAE_STRIDE_T - 1  # latent frame 0 is a single video frame
        self.chunk_video_frames.append(frames)
        if self.cuda:
            self.gpu_mem_allocated_gb.append(torch.cuda.memory_allocated() / GB)
            self.gpu_mem_reserved_gb.append(torch.cuda.memory_reserved() / GB)
        self.cpu_rss_gb.append(_cpu_rss_gb())
        self.retrieved_chunks.append(self._retrieval)
        self.bank_gb.append(self._bank)
        self._write_progress()

    def _write_progress(self):
        """This rank's memory so far, for runs that crash before save()."""
        progress = {
            "chunks_done": len(self.chunk_video_frames),
            "peak_gpu_mem_allocated_gb": torch.cuda.max_memory_allocated() / GB if self.cuda else None,
            "peak_gpu_mem_reserved_gb": torch.cuda.max_memory_reserved() / GB if self.cuda else None,
            "gpu_mem_reserved_gb": self.gpu_mem_reserved_gb,
            "cpu_rss_gb": self.cpu_rss_gb,
            "bank_gb": self.bank_gb,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.progress_path)), exist_ok=True)
            with open(self.progress_path, "w") as f:
                json.dump(progress, f)
        except OSError:
            pass  # progress is a fallback; never stop generation for it

    def _per_rank(self):
        """This rank's peaks and per-chunk CPU memory and bank, from every rank."""
        return _gather({
            "peak_alloc": torch.cuda.max_memory_allocated() / GB if self.cuda else None,
            "peak_reserved": torch.cuda.max_memory_reserved() / GB if self.cuda else None,
            "cpu_rss": self.cpu_rss_gb,
            "cpu_rss_start": self.cpu_rss_start_gb,
            "cpu_hwm": _proc_status_gb("VmHWM") if self.cpu_peak_reset else None,
            "bank": self.bank_gb,
        })

    def _read_events(self):
        """Turn the recorded CUDA markers into chunk times (once generation is done)."""
        if not self._events:
            return
        self._events[-1][1].synchronize()  # waits for the generation stream only
        self.chunk_times_s.extend(start.elapsed_time(end) / 1000 for start, end in self._events)
        self._events = []

    def summary(self):
        self._read_events()
        ranks = self._per_rank()
        fps = [f / t for f, t in zip(self.chunk_video_frames, self.chunk_times_s)]
        total_time = sum(self.chunk_times_s)
        cpu_peaks = [r["cpu_hwm"] for r in ranks]
        bank_total = _sum_per_chunk([r["bank"] for r in ranks])
        known_bank = [b for b in bank_total if b is not None]
        return {
            "num_chunks": len(self.chunk_times_s),
            "num_gpus": len(ranks),
            "gpu_name": torch.cuda.get_device_name() if self.cuda else None,
            "fps_last_chunk": fps[-1] if fps else None,
            "fps_mean": sum(self.chunk_video_frames) / total_time if total_time else None,
            "total_gen_time_s": total_time,
            "peak_gpu_mem_allocated_gb": [r["peak_alloc"] for r in ranks],
            "peak_gpu_mem_reserved_gb": [r["peak_reserved"] for r in ranks],
            # Sum of each process's own peak: processes may peak at different
            # times, so this can exceed the largest simultaneous total.
            "peak_cpu_rss_gb_per_rank": cpu_peaks,
            "peak_cpu_rss_total_gb": None if None in cpu_peaks else sum(cpu_peaks),
            "cpu_rss_start_total_gb": _sum_or_none([r["cpu_rss_start"] for r in ranks]),
            "peak_bank_gb": max(known_bank) if known_bank else None,
            "chunk_times_s": self.chunk_times_s,
            "chunk_video_frames": self.chunk_video_frames,
            "fps_per_chunk": fps,
            "gpu_mem_allocated_gb": self.gpu_mem_allocated_gb,
            "gpu_mem_reserved_gb": self.gpu_mem_reserved_gb,
            "cpu_rss_gb": self.cpu_rss_gb,  # rank 0
            "cpu_rss_total_gb": _sum_per_chunk([r["cpu_rss"] for r in ranks]),
            "bank_gb": bank_total,  # all ranks, right after each chunk is stored
            "retrieved_chunks": self.retrieved_chunks,
        }

    def save(self):
        """Write the JSON file. Call on every rank: peaks are gathered from all GPUs."""
        if not self.enabled:
            return
        summary = self.summary()
        try:
            os.remove(self.progress_path)  # the run finished, so the fallback is not needed
        except OSError:
            pass
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(summary, f, indent=1)
