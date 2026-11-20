import os

import pytest

from eval.experiments import natural_paths, predict


def test_predictions_match_the_smoke_test_measurements():
    # Smoke test run A (2026-11-17), 04_RLWSRL90_245: bank 35.2 -> 19.4 GB,
    # copies 142.6 -> 24.6 GB (20 -> 11 chunks stored, 81 -> 14 chunks copied).
    r = predict.predict(os.path.join(predict.BENCH_TRAJ, "RLWSRL90_245"), 245)
    assert (r["chunks"], r["bank_ours_chunks"]) == (20, 11)
    assert (r["copies_cpu_chunks"], r["copies_ours_chunks"]) == (81, 14)
    assert r["bank_ours_gb"] == pytest.approx(19.36, abs=0.01)
    assert r["copies_cpu_gb"] == pytest.approx(142.57, abs=0.01)


@pytest.mark.parametrize("name", list(natural_paths.SCENARIOS))
def test_ours_holds_and_copies_less_on_every_e3_path(name):
    folder = os.path.join(natural_paths.PATHS_DIR, f"{name}_{natural_paths.FRAMES}")
    r = predict.predict(folder, natural_paths.FRAMES)
    assert r["chunks"] == 54
    assert r["bank_ours_chunks"] < r["bank_cpu_chunks"]
    assert r["copies_ours_chunks"] < r["copies_cpu_chunks"]


def test_gpu_memory_lines():
    assert predict.first_out_of_memory("Full KV") == 28
    assert predict.first_out_of_memory("WorldKV (GPU)") == 32
    assert predict.first_out_of_memory("WorldKV (CPU)") is None
    assert predict.first_out_of_memory("Ours") is None
    assert predict.predict_gpu(26)["Full KV"] == pytest.approx(42.98)
