import numpy as np
import pytest
import torch

from eval import metrics

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def pair_metrics():
    return metrics.PairMetrics(CPU)


def random_frames(n, seed, size=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def test_identical_frames_score_perfectly(pair_metrics):
    a = metrics.to_tensor(random_frames(2, 0), CPU)
    s = pair_metrics(a, a.clone())
    assert np.all(np.isinf(s["psnr"]))
    assert np.allclose(s["ssim"], 1.0)
    assert np.allclose(s["lpips"], 0.0, atol=1e-6)


def test_unrelated_frames_score_badly(pair_metrics):
    # Noise vs. plain gray. (Two different noise images look alike to LPIPS,
    # since both are the same kind of texture.)
    a = metrics.to_tensor(random_frames(2, 0), CPU)
    b = torch.full_like(a, 0.5)
    s = pair_metrics(a, b)
    assert np.all(s["psnr"] < 12)  # uniform noise vs. mid-gray is ~10.8 dB
    assert np.all(s["ssim"] < 0.1)
    assert np.all(s["lpips"] > 0.3)


def test_pair_group():
    assert metrics.pair_group({"in_sink": True, "gap_latent": 50}) == "sink"
    assert metrics.pair_group({"in_sink": False, "gap_latent": 18}) == "short"
    assert metrics.pair_group({"in_sink": False, "gap_latent": 19}) == "long"


def make_pairs_result(num_frames, pairs):
    return {"trajectory": "test", "num_video_frames": num_frames, "pairs": [
        {"first": i, "revisit": j, "gap_frames": j - i,
         "gap_latent": (j - i) // 4, "in_sink": i < 9} for i, j in pairs]}


def test_evaluate_video_perfect_memory(pair_metrics):
    # Frames 10 and 11 repeat frames 0 and 1 exactly: perfect memory.
    video = random_frames(12, 0)
    video[10], video[11] = video[0], video[1]
    result = make_pairs_result(12, [(0, 10), (1, 11)])
    rows = metrics.evaluate_video(video, result, pair_metrics, batch_size=1)
    assert [r["group"] for r in rows] == ["sink", "sink"]
    assert all(np.isinf(r["psnr"]) and r["lpips"] < 1e-6 for r in rows)
    summary = metrics.summarize(rows)
    assert summary["all"]["n"] == 2 and summary["long"]["n"] == 0


def test_evaluate_video_rejects_wrong_length(pair_metrics):
    with pytest.raises(ValueError, match="frames"):
        metrics.evaluate_video(random_frames(5, 0), make_pairs_result(6, []), pair_metrics)


def test_set_metrics_identical_vs_different():
    same = metrics.SetMetrics(CPU, kid_subset_size=10)
    a = metrics.to_tensor(random_frames(20, 0), CPU)
    same.update(a, a.clone())
    diff = metrics.SetMetrics(CPU, kid_subset_size=10)
    flat = torch.full_like(a, 0.5)  # plain gray frames
    diff.update(a, flat)
    s, d = same.compute(), diff.compute()
    assert s["n"] == 20
    assert abs(s["fid"]) < 1e-3
    assert d["fid"] > 10 * max(s["fid"], 1.0)
