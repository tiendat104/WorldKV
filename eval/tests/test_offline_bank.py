import importlib.util
import os

import pytest
import torch

from eval.runner import REPO_ROOT
from eval.tests.test_retrieval_schedule import (CASES, CHUNK, NUM_RETRIEVE, RECENT_CHUNKS,
                                                SINK_CHUNKS, case_poses, rs)

# Loaded from its file: importing the wan package would load the model code.
_spec = importlib.util.spec_from_file_location(
    "offline_bank", os.path.join(REPO_ROOT, "wan", "offline_bank.py"))
ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ob)

LAYERS, TOKENS, HEADS, DIM = 3, 4, 2, 2


def fake_kv(chunk_id, layer):
    """One layer's stored K and V of a chunk, with values unique to (chunk, layer)."""
    k = torch.arange(TOKENS * HEADS * DIM, dtype=torch.float32).reshape(1, TOKENS, HEADS, DIM)
    k = k + 1000 * chunk_id + 100 * layer
    return k, -k


def test_copy_plan_reuses_chunks_already_on_the_gpu():
    assert ob.copy_plan([3, 5, 7, 8, 9], [5, 7, 8, 9, 11]) == [
        ("gpu", 1), ("gpu", 2), ("gpu", 3), ("gpu", 4), ("cpu", 11)]
    assert ob.copy_plan([], [2, 1]) == [("cpu", 2), ("cpu", 1)]


@pytest.mark.parametrize("source,name", CASES)
def test_prefetched_kv_equals_worldkv_concatenation(source, name):
    # Replays the loop of image2video_fast.py with --offline_schedule --prefetch.
    c2ws = case_poses(source, name)
    num_chunks = len(c2ws) // CHUNK
    schedule = rs.retrieval_schedule(c2ws, num_chunks, CHUNK, SINK_CHUNKS, RECENT_CHUNKS,
                                     NUM_RETRIEVE)
    last_use = rs.last_uses(schedule, num_chunks)
    free_after = rs.chunks_to_free(last_use)
    bank = ob.PrefetchBank(LAYERS, rs.max_live_chunks(last_use), NUM_RETRIEVE, "cpu")
    stored = 0
    for t in range(num_chunks):
        if schedule[t]:
            bank.use(schedule[t])
            new = [c for c in schedule[t] if c not in (schedule[t - 1] if t else [])]
            assert bank.cpu_bytes_last == len(new) * bank.chunk_bytes
            for layer in range(LAYERS):
                k, v = bank.retrieved(layer)
                # What WorldKV gives the model: the chunks concatenated in retrieval order.
                expected_k = torch.cat([fake_kv(c, layer)[0] for c in schedule[t]], dim=1)
                assert torch.equal(k, expected_k) and torch.equal(v, -expected_k)
                assert k.shape == expected_k.shape
        if t + 1 < num_chunks:
            bank.prefetch(schedule[t + 1])
        if last_use[t] >= 0:
            for layer in range(LAYERS):
                bank.store(t, layer, *fake_kv(t, layer))
            stored += 1
        bank.free(free_after[t])
        assert len(bank.slot_of) <= bank.num_slots
    assert stored > 0 and bank.slot_of == {}
    bank.close()


def test_use_and_store_fail_loudly():
    bank = ob.PrefetchBank(LAYERS, 1, NUM_RETRIEVE, "cpu")
    for layer in range(LAYERS):
        bank.store(1, layer, *fake_kv(1, layer))
    with pytest.raises(RuntimeError, match="no free slot"):
        bank.store(2, 0, *fake_kv(2, 0))
    bank.prefetch([1])
    with pytest.raises(RuntimeError, match="prefetched"):
        bank.use([2])


def test_max_live_chunks():
    # Chunk 0 never used; 1 used until 4; 2 until 3; 3 until 5.
    assert rs.max_live_chunks([-1, 4, 3, 5, -1, -1]) == 3


def test_generation_loop_has_the_prefetch_steps():
    with open(os.path.join(REPO_ROOT, "wan", "image2video_fast.py")) as f:
        src = f.read()
    for step in ("prefetch_bank.use(selected_ids)", "prefetch_bank.retrieved(layer_idx)",
                 "prefetch_bank.prefetch(schedule[chunk_id + 1])", "prefetch_bank.store(",
                 "prefetch_bank.free(free_after[chunk_id])", "prefetch_bank.close()",
                 "telemetry.start_retrieval()", "telemetry.end_retrieval(copied_bytes)"):
        assert src.count(step) == 1, step
