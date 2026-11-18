"""Which past chunks WorldKV retrieves for each chunk, from camera poses alone.

`select_chunks` is the selection used by the generation loop in
image2video_fast.py. It reads only the absolute latent camera poses, never the
KV bank, so with a known camera path the whole schedule can be computed before
generation (`retrieval_schedule`), and with it the last chunk that needs each
stored chunk (`last_uses`). The offline mode frees each stored chunk right after
that last use (`chunks_to_free`, `free_chunks`).

Only torch is imported, so this file can be loaded without the model code.
"""

import torch


def select_chunks(c2ws_abs, chunk_id, chunk_size, sink_chunks, recent_chunks,
                  num_retrieve_chunks, bank_len):
    """Ids of the stored chunks retrieved for chunk `chunk_id`, closest first.

    Args:
        c2ws_abs: absolute camera-to-world poses, one per latent frame [N, 4, 4].
        chunk_id: the chunk about to be generated.
        sink_chunks, recent_chunks: chunks excluded from the candidates (the
            sink at the start, the most recent ones before `chunk_id`).
        num_retrieve_chunks: how many chunks to retrieve.
        bank_len: number of chunks stored in the bank so far.

    Returns:
        A list of chunk ids, in the order they are concatenated (empty if there
        is no candidate).
    """
    # Current chunk average pose (absolute)
    cs = chunk_id * chunk_size
    ce = min(cs + chunk_size, c2ws_abs.shape[0])
    current_trans = c2ws_abs[cs:ce, :3, 3].mean(dim=0)
    current_rot = c2ws_abs[cs:ce, :3, :3].mean(dim=0)

    # Candidate chunks: exclude sink and recent
    recent_start_chunk = max(sink_chunks, chunk_id - recent_chunks)
    candidate_ids = []
    candidate_trans = []
    candidate_rots = []
    for cid in range(sink_chunks, recent_start_chunk):
        if cid < bank_len:
            s = cid * chunk_size
            e = min(s + chunk_size, c2ws_abs.shape[0])
            candidate_ids.append(cid)
            candidate_trans.append(c2ws_abs[s:e, :3, 3].mean(dim=0))
            candidate_rots.append(c2ws_abs[s:e, :3, :3].mean(dim=0))

    if not (candidate_ids and num_retrieve_chunks > 0):
        return []

    # Translation L2 distance
    candidate_trans = torch.stack(candidate_trans).to(current_trans.device)
    trans_dist = ((candidate_trans - current_trans) ** 2).sum(dim=-1)

    # Rotation geodesic distance: arccos((trace(R1^T @ R2) - 1) / 2)
    candidate_rots = torch.stack(candidate_rots).to(current_rot.device)
    rel_rot = candidate_rots.transpose(-1, -2) @ current_rot.unsqueeze(0)
    trace_val = rel_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [N]
    cos_angle = (trace_val - 1.0) / 2.0
    rot_dist = torch.acos(cos_angle.clamp(-1.0, 1.0))  # [N], radians

    # Normalize each to [0, 1] then combine
    trans_norm = trans_dist / trans_dist.max() if trans_dist.max() > 0 else trans_dist
    rot_norm = rot_dist / rot_dist.max() if rot_dist.max() > 0 else rot_dist
    distances = 0.5 * trans_norm + 0.5 * rot_norm

    k = min(num_retrieve_chunks, len(candidate_ids))
    _, top_idx = distances.topk(k, largest=False)
    return [candidate_ids[i] for i in top_idx.tolist()]


def retrieval_schedule(c2ws_abs, num_chunks, chunk_size, sink_chunks, recent_chunks,
                       num_retrieve_chunks):
    """`select_chunks` for every chunk of the video.

    The generation loop stores one chunk per step, so when chunk `t` is
    generated the bank holds chunks 0 .. t-1.
    """
    return [select_chunks(c2ws_abs, t, chunk_size, sink_chunks, recent_chunks,
                          num_retrieve_chunks, bank_len=t)
            for t in range(num_chunks)]


def last_uses(schedule, num_chunks):
    """For each chunk, the last chunk that retrieves it, or -1 if none does."""
    last = [-1] * num_chunks
    for t, selected_ids in enumerate(schedule):
        for cid in selected_ids:
            last[cid] = max(last[cid], t)
    return last


def max_live_chunks(last_use):
    """Most chunks the offline bank holds at once.

    Chunk t is stored before the chunks last used at t are freed, so right
    after storing chunk t the bank holds every chunk c <= t with last use >= t.
    """
    return max(sum(1 for c in range(t + 1) if last_use[c] >= t)
               for t in range(len(last_use)))


def chunks_to_free(last_use):
    """For each chunk t, the stored chunks to free once chunk t is generated."""
    free_after = [[] for _ in last_use]
    for cid, t in enumerate(last_use):
        if t >= 0:
            free_after[t].append(cid)
    return free_after


def free_chunks(kv_bank_k, kv_bank_v, chunk_ids):
    """Drop the KV of `chunk_ids` in every layer.

    Entries become None instead of being removed, so that the list index stays
    the chunk id (select_chunks and the retrieval loop rely on it).
    """
    for layer_k, layer_v in zip(kv_bank_k, kv_bank_v):
        for cid in chunk_ids:
            layer_k[cid] = None
            layer_v[cid] = None


def check_retrieval(chunk_id, selected_ids, planned_ids, bank_layer):
    """Fail loudly if the loop's choice differs from the plan or needs a freed chunk."""
    if selected_ids != planned_ids:
        raise RuntimeError(f"chunk {chunk_id}: retrieved {selected_ids}, "
                           f"but the offline schedule planned {planned_ids}")
    missing = [cid for cid in selected_ids if bank_layer[cid] is None]
    if missing:
        raise RuntimeError(f"chunk {chunk_id}: chunks {missing} were freed but are needed")


def entry_bytes(kv_bank_k, kv_bank_v, chunk_ids):
    """Bytes of the given chunks over all layers, K and V (what a retrieval copies)."""
    return sum(layer[cid].numel() * layer[cid].element_size()
               for layer in kv_bank_k + kv_bank_v for cid in chunk_ids)


def bank_bytes(kv_bank_k, kv_bank_v):
    """Bytes held by the bank's KV tensors (freed entries are None)."""
    return sum(t.numel() * t.element_size()
               for layer in kv_bank_k + kv_bank_v for t in layer if t is not None)
