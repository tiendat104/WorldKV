import json
import os
import shutil

import imageio.v3 as iio
import numpy as np

from eval import check_offline

RUN = "s0_T_100"
RETRIEVED = [[], [], [], [], [1], [2, 1]]  # chunks 1 and 2 last used at chunk 5
CHUNK_GB = 0.5


def write_run(out, method, retrieved, bank_gb, frames):
    os.makedirs(os.path.join(out, "runs", method))
    with open(os.path.join(out, "runs", method, RUN + ".telemetry.json"), "w") as f:
        json.dump({"num_chunks": len(retrieved), "retrieved_chunks": retrieved,
                   "bank_gb": bank_gb, "peak_bank_gb": max(bank_gb),
                   "peak_gpu_mem_reserved_gb": [30.0, 31.0], "copied_total_gb": 3.0,
                   "retrieval_s": [None] * 4 + [0.5, 0.7], "chunk_times_s": [24.0] * 6,
                   "fps_mean": 0.5}, f)
    os.makedirs(os.path.join(out, "videos", method))
    iio.imwrite(os.path.join(out, "videos", method, RUN + ".mp4"), frames, fps=16)


def frames(seed):
    return np.random.default_rng(seed).integers(0, 256, (8, 32, 48, 3), dtype=np.uint8)


def make_out(tmp_path, offline_bank=None, offline_frames=None, offline_retrieved=RETRIEVED):
    out = str(tmp_path / "out")
    write_run(out, "cpu_only", RETRIEVED, [(t + 1) * CHUNK_GB for t in range(6)], frames(0))
    bank = offline_bank or [n * CHUNK_GB for n in [0, 1, 2, 2, 2, 2]]
    write_run(out, "offline", offline_retrieved, bank,
              frames(0) if offline_frames is None else offline_frames)
    return out


def test_live_chunks():
    assert check_offline.live_chunks(RETRIEVED, 6) == [0, 1, 2, 2, 2, 2]


def test_all_checks_pass_for_a_matching_run(tmp_path):
    out = make_out(tmp_path)
    reference = str(tmp_path / "reference.mp4")
    shutil.copy(os.path.join(out, "videos", "cpu_only", RUN + ".mp4"), reference)
    results, rows = check_offline.check(out, RUN, reference)
    assert len(results) == 4 and all(ok for ok, _ in results), results
    assert dict((name, (c, o)) for name, c, o in rows)["peak bank (GB, all GPUs)"] == (3.0, 1.0)


def test_failures_are_reported(tmp_path):
    out = make_out(tmp_path, offline_bank=[0.0, 0.5, 1.0, 1.5, 1.0, 1.0],
                   offline_frames=frames(1), offline_retrieved=[[], [], [], [], [1], [1, 2]])
    results = dict((msg.split(" (")[0], ok) for ok, msg in check_offline.check(out, RUN)[0])
    assert results == {"same retrieved chunks at every chunk": False,
                       "offline video equals cpu_only video": False,
                       "offline bank = chunks still needed x 0.500 GB per chunk": False}


def test_offline_only_run(tmp_path):
    out = str(tmp_path / "out")
    write_run(out, "offline", RETRIEVED, [n * CHUNK_GB for n in [0, 1, 2, 2, 2, 2]], frames(0))
    results, rows = check_offline.check(out, RUN)
    assert results == [] and all(c is None for _, c, _ in rows)
