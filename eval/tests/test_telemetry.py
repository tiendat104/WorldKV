import json
import os
import time

import pytest

from eval import telemetry
from eval.runner import REPO_ROOT


def test_disabled_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(telemetry.ENV_VAR, raising=False)
    t = telemetry.ChunkTelemetry.from_env()
    assert not t.enabled
    t.start_chunk()
    t.end_chunk(3)
    t.save()
    assert t.chunk_times_s == []


def test_records_chunks_and_saves(monkeypatch, tmp_path):
    path = tmp_path / "run.telemetry.json"
    monkeypatch.setenv(telemetry.ENV_VAR, str(path))
    t = telemetry.ChunkTelemetry.from_env()
    for _ in range(3):
        t.start_chunk()
        time.sleep(0.02)
        t.end_chunk(3)
    t.save()

    saved = json.loads(path.read_text())
    # First chunk: 1 + 4 + 4 = 9 video frames; later chunks: 3 x 4 = 12.
    assert saved["chunk_video_frames"] == [9, 12, 12]
    assert saved["num_chunks"] == 3
    assert all(s >= 0.02 for s in saved["chunk_times_s"])
    assert abs(saved["fps_last_chunk"] - 12 / saved["chunk_times_s"][-1]) < 1e-9
    assert abs(saved["fps_mean"] - 33 / sum(saved["chunk_times_s"])) < 1e-9
    assert saved["num_gpus"] == 1


class FakeEvent:
    """Stands in for torch.cuda.Event: each record() advances a clock by 0.5 s."""
    clock = 0.0

    def __init__(self, enable_timing=False):
        self.stream, self.time, self.waited = None, None, False

    def record(self, stream):
        FakeEvent.clock += 0.5
        self.stream, self.time = stream, FakeEvent.clock

    def synchronize(self):
        self.waited = True

    def elapsed_time(self, end):
        return (end.time - self.time) * 1000  # milliseconds, like CUDA


@pytest.fixture
def fake_gpu(monkeypatch):
    cuda = telemetry.torch.cuda

    def no_synchronize():
        raise AssertionError("torch.cuda.synchronize() would stall the other streams")

    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "synchronize", no_synchronize)
    monkeypatch.setattr(cuda, "Event", FakeEvent)
    monkeypatch.setattr(cuda, "current_stream", lambda *a, **k: "current")
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(cuda, "get_device_name", lambda *a, **k: "fake")
    for name in ("memory_allocated", "memory_reserved",
                 "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(cuda, name, lambda *a, **k: 0)


def test_gpu_times_come_from_events_without_waiting(fake_gpu, tmp_path):
    path = tmp_path / "run.telemetry.json"
    t = telemetry.ChunkTelemetry(str(path), stream="generation")
    for _ in range(3):
        t.start_chunk()
        t.end_chunk(3)
    # Nothing is read (so nothing waits) until the run ends.
    assert t.chunk_times_s == []
    assert all(e.stream == "generation" for pair in t._events for e in pair)
    last_end = t._events[-1][1]
    t.save()
    assert last_end.waited  # only the generation stream's last marker
    saved = json.loads(path.read_text())
    assert saved["chunk_times_s"] == pytest.approx([0.5, 0.5, 0.5])
    assert saved["fps_last_chunk"] == pytest.approx(12 / 0.5)


def test_gpu_markers_default_to_current_stream(fake_gpu, tmp_path):
    t = telemetry.ChunkTelemetry(str(tmp_path / "run.telemetry.json"))
    t.start_chunk()
    t.end_chunk(3)
    assert [e.stream for e in t._events[0]] == ["current", "current"]


def _distributed_worker(rank, world_size, path, port):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      **{telemetry.ENV_VAR: path})
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    t = telemetry.ChunkTelemetry.from_env()
    t.start_chunk()
    t.end_chunk(3)
    t.save()  # every rank calls save(); only rank 0 writes
    dist.destroy_process_group()


def test_save_with_several_processes(tmp_path):
    # Like torchrun with one process per GPU, but on CPU with 2 processes.
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    path = tmp_path / "run.telemetry.json"
    mp.spawn(_distributed_worker, args=(2, str(path), port), nprocs=2)
    saved = json.loads(path.read_text())
    assert saved["num_gpus"] == 2
    assert len(saved["peak_gpu_mem_allocated_gb"]) == 2


def test_hooks_present_in_worldkv():
    # Guards against losing the hooks, e.g. when merging with other branches.
    with open(os.path.join(REPO_ROOT, "wan", "image2video_fast.py")) as f:
        src = f.read()
    for hook in ("ChunkTelemetry.from_env()", "telemetry.start_chunk()",
                 "telemetry.end_chunk(chunk_size)", "telemetry.save()"):
        assert src.count(hook) == 1, hook
