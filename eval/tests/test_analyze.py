import json
import os

import pytest

from eval.experiments import analyze


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def telemetry(chunks, fps, gpu, bank=None, rss=None, copied=None, wait=None, chunk_peaks=None):
    return {"num_chunks": chunks, "fps_mean": fps, "chunk_times_s": [12 / fps] * chunks,
            "retrieval_s": [None] * 4 + [wait] * (chunks - 4), "copied_total_gb": copied,
            "peak_gpu_mem_reserved_gb": [gpu - 0.5, gpu], "peak_bank_gb": bank,
            "peak_cpu_rss_total_gb": rss,
            "chunk_peak_gpu_mem_reserved_gb": chunk_peaks or [gpu] * chunks}


def make_results(root):
    """A results folder like run_experiments.sh writes, with made-up numbers."""
    e1 = os.path.join(root, "e1")
    checks = []
    for run in analyze.run_names("e1.json"):
        write(f"{e1}/runs/gpu_only/{run}.telemetry.json", telemetry(20, 0.48, 39.3))
        write(f"{e1}/runs/cpu_only/{run}.telemetry.json",
              telemetry(20, 0.38, 30.9, copied=142.6, wait=0.53))
        write(f"{e1}/runs/offline/{run}.telemetry.json",
              telemetry(20, 0.39, 33.4, copied=24.6, wait=0.0))
        checks += [f"--- {run}", "PASS same retrieved chunks at every chunk",
                   "PASS offline video equals cpu_only video (same file bytes)",
                   "PASS offline bank = chunks still needed x 1.760 GB per chunk",
                   f"PASS offline video equals {run}.mp4 (same decoded frames)"]
    with open(f"{e1}/checks.txt", "w") as f:
        f.write("\n".join(checks) + "\n")

    long_dir, long_run = os.path.join(root, "e2_long"), analyze.run_names("e2_long.json")[0]
    # WorldKV (GPU) ran out of memory after 31 chunks: only its progress files remain.
    for rank in range(2):
        write(f"{long_dir}/runs/gpu_only/{long_run}.telemetry.json.rank{rank}.progress.json",
              {"chunks_done": 31,
               "chunk_peak_gpu_mem_reserved_gb": [30 + 0.45 * k + 0.1 * rank for k in range(31)]})
    write(f"{long_dir}/runs/gpu_only/{long_run}.json",
          {"status": "failed", "out_of_memory": True, "chunks_done": 31})
    write(f"{long_dir}/runs/cpu_only/{long_run}.telemetry.json", telemetry(40, 0.38, 30.9))
    write(f"{long_dir}/runs/offline/{long_run}.telemetry.json", telemetry(40, 0.39, 34.4))

    full_dir = os.path.join(root, "e2_full_kv")
    for run in analyze.run_names("e2_full_kv.json"):
        chunks = (int(run.rsplit("_", 1)[1]) + 3) // 12
        if chunks <= 27:
            write(f"{full_dir}/runs/full_kv/{run}.telemetry.json",
                  telemetry(chunks, 0.4, 19.7 + 0.9 * chunks))
        else:
            write(f"{full_dir}/runs/full_kv/{run}.json",
                  {"status": "failed", "out_of_memory": True, "chunks_done": 0})

    e3 = os.path.join(root, "e3")
    for run in analyze.run_names("e3.json"):
        write(f"{e3}/runs/cpu_only/{run}.telemetry.json",
              telemetry(54, 0.38, 30.9, bank=95.0, rss=103.0, copied=501.6, wait=0.55))
        write(f"{e3}/runs/offline/{run}.telemetry.json",
              telemetry(54, 0.39, 34.4, bank=44.0, rss=52.3, copied=147.8, wait=0.0))
    return root


def test_e1_tables(tmp_path):
    root = make_results(str(tmp_path))
    text = analyze.e1_tables(os.path.join(root, "e1"))
    assert "| Ours | 4 | 0.390 | 30.77 | 0.000 | 24.6 | 33.4 |" in text
    assert "| WorldKV (CPU) | 4 | 0.380 | 31.58 | 0.530 | 142.6 | 30.9 |" in text
    assert "| WorldKV (GPU) | 4 | 0.480 | 25.00 | - | - | 39.3 |" in text  # no CPU->GPU copies
    assert text.count("| Ours | yes |") == 4  # identical to both baselines


def test_e2_data(tmp_path):
    root = make_results(str(tmp_path))
    data = analyze.e2_data(os.path.join(root, "e2_long"), os.path.join(root, "e2_full_kv"))
    gpu = data["gpu_only"]
    # From the progress files: chunks 9..31, largest of the 2 GPUs; out of memory at 32.
    assert gpu["points"][0] == (pytest.approx(analyze.seconds(9)), pytest.approx(30 + 0.45 * 8 + 0.1))
    assert len(gpu["points"]) == 31 - 8
    assert gpu["out_of_memory"] == pytest.approx(analyze.seconds(32))
    assert data["offline"]["out_of_memory"] is None and len(data["offline"]["points"]) == 32
    full = data["full_kv"]
    assert [round(x, 1) for x, _ in full["points"]] == [6.6, 13.3, 19.3, 20.1]
    assert full["out_of_memory"] == pytest.approx(analyze.seconds(28))
    assert "First out of memory" in analyze.e2_table(data)


def test_e3_table(tmp_path):
    root = make_results(str(tmp_path))
    text = analyze.e3_table(os.path.join(root, "e3"))
    for path in ("inspect", "tour", "return", "street"):
        assert f"| {path} |" in text
    assert "| 95.0 -> 44.0 | 54% | 103.0 -> 52.3 | 49% |" in text


def test_main_writes_the_tables(tmp_path, monkeypatch):
    root = make_results(str(tmp_path))
    monkeypatch.setattr("sys.argv", ["analyze", root])
    analyze.main()
    for name in ("e1.md", "e2.md", "e3.md"):
        assert os.path.exists(os.path.join(root, "analysis", name))


def test_e2_figure(tmp_path):
    pytest.importorskip("matplotlib")
    root = make_results(str(tmp_path))
    data = analyze.e2_data(os.path.join(root, "e2_long"), os.path.join(root, "e2_full_kv"))
    assert analyze.plot_e2(data, os.path.join(root, "e2"))
    assert os.path.exists(os.path.join(root, "e2.png"))
