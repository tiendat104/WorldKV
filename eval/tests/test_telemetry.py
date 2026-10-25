import json
import os
import time

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
