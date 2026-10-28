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
  - CPU memory (RSS) of the process per chunk, e.g. for the CPU-side bank.

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


def _cpu_rss_gb():
    """Current resident memory of this process in GB, or None if unknown."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / GB
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return None


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
        self._start = None
        self._events = []  # (start, end) CUDA events per chunk, read in save()
        if self.enabled and self.cuda:
            # Peaks count from here: model weights already loaded are included.
            torch.cuda.reset_peak_memory_stats()

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

    def _per_rank_peaks(self):
        """Peak GPU memory of every rank, as lists (one value per GPU)."""
        mine = [torch.cuda.max_memory_allocated() / GB,
                torch.cuda.max_memory_reserved() / GB] if self.cuda else [None, None]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
        return [g[0] for g in gathered], [g[1] for g in gathered]

    def _read_events(self):
        """Turn the recorded CUDA markers into chunk times (once generation is done)."""
        if not self._events:
            return
        self._events[-1][1].synchronize()  # waits for the generation stream only
        self.chunk_times_s.extend(start.elapsed_time(end) / 1000 for start, end in self._events)
        self._events = []

    def summary(self):
        self._read_events()
        peak_alloc, peak_reserved = self._per_rank_peaks()
        fps = [f / t for f, t in zip(self.chunk_video_frames, self.chunk_times_s)]
        total_time = sum(self.chunk_times_s)
        return {
            "num_chunks": len(self.chunk_times_s),
            "num_gpus": len(peak_alloc),
            "gpu_name": torch.cuda.get_device_name() if self.cuda else None,
            "fps_last_chunk": fps[-1] if fps else None,
            "fps_mean": sum(self.chunk_video_frames) / total_time if total_time else None,
            "total_gen_time_s": total_time,
            "peak_gpu_mem_allocated_gb": peak_alloc,
            "peak_gpu_mem_reserved_gb": peak_reserved,
            "chunk_times_s": self.chunk_times_s,
            "chunk_video_frames": self.chunk_video_frames,
            "fps_per_chunk": fps,
            "gpu_mem_allocated_gb": self.gpu_mem_allocated_gb,
            "gpu_mem_reserved_gb": self.gpu_mem_reserved_gb,
            "cpu_rss_gb": self.cpu_rss_gb,
        }

    def save(self):
        """Write the JSON file. Call on every rank: peaks are gathered from all GPUs."""
        if not self.enabled:
            return
        summary = self.summary()
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(summary, f, indent=1)
