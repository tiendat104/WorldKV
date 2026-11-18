import importlib.util
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eval import pairing
from eval.runner import REPO_ROOT
from eval.tests.test_pairing import BENCH_TRAJ, worldkv_latent_poses

# Loaded from its file: importing the wan package would load the model code.
_spec = importlib.util.spec_from_file_location(
    "retrieval_schedule", os.path.join(REPO_ROOT, "wan", "retrieval_schedule.py"))
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

# WorldKV's window in our runs (eval/runner.py WORLDKV_FLAGS): sink 3 latent
# frames (1 chunk), 18 retrieval frames (6 chunks), recent 6 frames (2 chunks).
CHUNK, SINK_CHUNKS, RECENT_CHUNKS, NUM_RETRIEVE = 3, 1, 2, 6
TRAJECTORIES = sorted(os.listdir(BENCH_TRAJ))


def original_selection(c2ws_infer_abs, chunk_id, bank_len, chunk_size=CHUNK,
                       sink_chunks=SINK_CHUNKS, recent_chunks=RECENT_CHUNKS,
                       num_retrieve_chunks=NUM_RETRIEVE):
    """The selection code of wan/image2video_fast.py before it moved to
    select_chunks (commit 4731a80), with len(kv_bank_k[0]) as `bank_len`."""
    cs = chunk_id * chunk_size
    ce = min(cs + chunk_size, c2ws_infer_abs.shape[0])
    current_trans = c2ws_infer_abs[cs:ce, :3, 3].mean(dim=0)
    current_rot = c2ws_infer_abs[cs:ce, :3, :3].mean(dim=0)
    recent_start_chunk = max(sink_chunks, chunk_id - recent_chunks)
    candidate_ids = []
    candidate_trans = []
    candidate_rots = []
    for cid in range(sink_chunks, recent_start_chunk):
        if cid < bank_len:
            s = cid * chunk_size
            e = min(s + chunk_size, c2ws_infer_abs.shape[0])
            candidate_ids.append(cid)
            candidate_trans.append(c2ws_infer_abs[s:e, :3, 3].mean(dim=0))
            candidate_rots.append(c2ws_infer_abs[s:e, :3, :3].mean(dim=0))
    if candidate_ids and num_retrieve_chunks > 0:
        candidate_trans = torch.stack(candidate_trans).to(current_trans.device)
        trans_dist = ((candidate_trans - current_trans) ** 2).sum(dim=-1)
        candidate_rots = torch.stack(candidate_rots).to(current_rot.device)
        rel_rot = candidate_rots.transpose(-1, -2) @ current_rot.unsqueeze(0)
        trace_val = rel_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cos_angle = (trace_val - 1.0) / 2.0
        rot_dist = torch.acos(cos_angle.clamp(-1.0, 1.0))
        trans_norm = trans_dist / trans_dist.max() if trans_dist.max() > 0 else trans_dist
        rot_norm = rot_dist / rot_dist.max() if rot_dist.max() > 0 else rot_dist
        distances = 0.5 * trans_norm + 0.5 * rot_norm
        k = min(num_retrieve_chunks, len(candidate_ids))
        _, top_idx = distances.topk(k, largest=False)
        return [candidate_ids[i] for i in top_idx.tolist()]
    return []


def benchmark_poses(name):
    """Latent poses of a benchmark trajectory, as generation computes them."""
    poses = pairing.load_poses(os.path.join(BENCH_TRAJ, name))
    frame_num = int(name.rsplit("_", 1)[1])
    return torch.from_numpy(worldkv_latent_poses(poses, frame_num))


def synthetic_poses(kind, num_frames=120):
    rng = np.random.default_rng(0)
    c2ws = np.tile(np.eye(4), (num_frames, 1, 1))
    if kind == "random":
        c2ws[:, :3, :3] = Rotation.random(num_frames, random_state=1).as_matrix()
        c2ws[:, :3, 3] = rng.normal(size=(num_frames, 3))
    elif kind == "pauses":
        # Turn 30 frames, stand still 30 frames, turn back, stand still: many ties.
        angles = np.concatenate([np.linspace(0, 90, 30), np.full(30, 90.0),
                                 np.linspace(90, 0, 30), np.zeros(30)])
        # One column of angles: newer SciPy (1.17) rejects a flat array here.
        c2ws[:, :3, :3] = Rotation.from_euler("y", angles[:, None], degrees=True).as_matrix()
    # "still": every pose is the identity, so every distance is 0.
    return torch.from_numpy(c2ws).float()


CASES = ([("benchmark", name) for name in TRAJECTORIES]
         + [("synthetic", kind) for kind in ("random", "pauses", "still")])


def case_poses(source, name):
    return benchmark_poses(name) if source == "benchmark" else synthetic_poses(name)


@pytest.mark.parametrize("source,name", CASES)
def test_select_chunks_matches_original_code(source, name):
    # Same chunks in the same order (the order changes floating-point sums).
    c2ws = case_poses(source, name)
    for t in range(len(c2ws) // CHUNK):
        assert rs.select_chunks(c2ws, t, CHUNK, SINK_CHUNKS, RECENT_CHUNKS, NUM_RETRIEVE,
                                bank_len=t) == original_selection(c2ws, t, bank_len=t), t


@pytest.mark.parametrize("source,name", CASES)
def test_schedule_respects_the_window(source, name):
    c2ws = case_poses(source, name)
    num_chunks = len(c2ws) // CHUNK
    schedule = rs.retrieval_schedule(c2ws, num_chunks, CHUNK, SINK_CHUNKS, RECENT_CHUNKS,
                                     NUM_RETRIEVE)
    assert len(schedule) == num_chunks
    for t, selected in enumerate(schedule):
        candidates = range(SINK_CHUNKS, max(SINK_CHUNKS, t - RECENT_CHUNKS))
        assert len(set(selected)) == len(selected) == min(NUM_RETRIEVE, len(candidates))
        assert all(cid in candidates for cid in selected)


def test_last_uses():
    schedule = [[], [], [0], [1, 0], [2]]
    assert rs.last_uses(schedule, 5) == [3, 3, 4, -1, -1]


@pytest.mark.parametrize("name", TRAJECTORIES)
def test_sink_and_last_two_chunks_are_never_retrieved(name):
    c2ws = benchmark_poses(name)
    num_chunks = len(c2ws) // CHUNK
    last = rs.last_uses(rs.retrieval_schedule(c2ws, num_chunks, CHUNK, SINK_CHUNKS,
                                              RECENT_CHUNKS, NUM_RETRIEVE), num_chunks)
    assert last[0] == last[-1] == last[-2] == -1


def run_bank(c2ws, offline, num_layers=2):
    """The bank steps of image2video_fast.py's loop, with tiny fake KV tensors.

    Each stored tensor is filled with its chunk id, so a retrieval can check that
    it got the right data. Returns the chunks retrieved per step and the bank's
    bytes after each step.
    """
    num_chunks = len(c2ws) // CHUNK
    kv_bank_k = [[] for _ in range(num_layers)]
    kv_bank_v = [[] for _ in range(num_layers)]
    if offline:
        schedule = rs.retrieval_schedule(c2ws, num_chunks, CHUNK, SINK_CHUNKS,
                                         RECENT_CHUNKS, NUM_RETRIEVE)
        last_use = rs.last_uses(schedule, num_chunks)
        free_after = rs.chunks_to_free(last_use)
    retrieved, bank_bytes = [], []
    for chunk_id in range(num_chunks):
        selected_ids = []
        if len(kv_bank_k[0]) > 0:
            selected_ids = rs.select_chunks(c2ws, chunk_id, CHUNK, SINK_CHUNKS, RECENT_CHUNKS,
                                            NUM_RETRIEVE, bank_len=len(kv_bank_k[0]))
            if offline:
                rs.check_retrieval(chunk_id, selected_ids, schedule[chunk_id], kv_bank_k[0])
            for layer in kv_bank_k + kv_bank_v:
                got = torch.cat([layer[i] for i in selected_ids], dim=1) if selected_ids else None
                if got is not None:
                    assert got[0, :, 0, 0].tolist() == [float(i) for i in selected_ids]
        retrieved.append(selected_ids)
        for layer_idx in range(num_layers):
            if offline and last_use[chunk_id] < 0:
                kv_bank_k[layer_idx].append(None)
                kv_bank_v[layer_idx].append(None)
                continue
            kv_bank_k[layer_idx].append(torch.full((1, 1, 2, 4), float(chunk_id)))
            kv_bank_v[layer_idx].append(torch.full((1, 1, 2, 4), float(chunk_id)))
        if offline:
            rs.free_chunks(kv_bank_k, kv_bank_v, free_after[chunk_id])
        bank_bytes.append(rs.bank_bytes(kv_bank_k, kv_bank_v))
    return retrieved, bank_bytes


@pytest.mark.parametrize("source,name", CASES)
def test_offline_bank_retrieves_the_same_chunks_and_holds_only_live_ones(source, name):
    c2ws = case_poses(source, name)
    num_chunks = len(c2ws) // CHUNK
    full_retrieved, full_bytes = run_bank(c2ws, offline=False)
    offline_retrieved, offline_bytes = run_bank(c2ws, offline=True)
    assert offline_retrieved == full_retrieved
    # After chunk t, the bank holds exactly the stored chunks still needed later.
    last_use = rs.last_uses(full_retrieved, num_chunks)
    entry_bytes = full_bytes[0]  # one chunk: 2 layers x (K + V)
    for t in range(num_chunks):
        live = sum(1 for cid in range(t + 1) if last_use[cid] > t)
        assert offline_bytes[t] == live * entry_bytes
        assert full_bytes[t] == (t + 1) * entry_bytes
    assert offline_bytes[-1] == 0


def test_check_retrieval_fails_loudly():
    bank_layer = [torch.zeros(1), None, torch.zeros(1)]
    rs.check_retrieval(5, [2, 0], [2, 0], bank_layer)
    with pytest.raises(RuntimeError, match="planned"):
        rs.check_retrieval(5, [0, 2], [2, 0], bank_layer)
    with pytest.raises(RuntimeError, match="freed"):
        rs.check_retrieval(5, [1], [1], bank_layer)


def test_free_chunks_keeps_positions():
    kv_bank_k = [[torch.zeros(2), torch.zeros(2), torch.zeros(2)] for _ in range(3)]
    kv_bank_v = [[torch.zeros(2), torch.zeros(2), torch.zeros(2)] for _ in range(3)]
    rs.free_chunks(kv_bank_k, kv_bank_v, [1])
    for layer in kv_bank_k + kv_bank_v:
        assert len(layer) == 3 and layer[1] is None and layer[0] is not None
    assert rs.bank_bytes(kv_bank_k, kv_bank_v) == 6 * 2 * 2 * 4  # 6 layers x 2 chunks x 2 floats
    assert rs.chunks_to_free([3, 3, -1, 4, -1]) == [[], [], [], [0, 1], [3]]


def test_generation_loop_uses_select_chunks():
    # Guards against the loop getting its own copy of the selection again.
    with open(os.path.join(REPO_ROOT, "wan", "image2video_fast.py")) as f:
        src = f.read()
    assert src.count("select_chunks(") == 1
    assert "candidate_trans" not in src
