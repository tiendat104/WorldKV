"""KV bank for the offline schedule with prefetch (CPU slots + two GPU buffers).

With a known retrieval schedule, the chunks retrieved for chunk t+1 can be
copied to the GPU while chunk t is generated:

  - The CPU bank is one pinned pool of slots, allocated at the first store and
    sized by the most chunks the schedule ever needs at once. A freed chunk's
    slot is given to a later chunk, so CPU memory is fixed and known in
    advance. The pool is pinned with cudaHostRegister (exact size; PyTorch's
    pinned allocator would round it up), which background copies need.
  - Two GPU buffers (A and B) each hold the retrieved chunks of one chunk, per
    layer and K/V, in retrieval order. While chunk t reads one buffer, the
    other is filled for chunk t+1 on a separate CUDA stream: chunks that chunk
    t also retrieved are copied from the current buffer (GPU to GPU), only new
    ones come from the CPU slots. Then the two swap.

The retrieved KV given to the model holds the same values in the same order as
WorldKV's torch.cat of bank entries, so the video does not change.

Only torch is imported, so this file can be loaded without the model code.
"""

from contextlib import nullcontext

import torch


def copy_plan(current_ids, next_ids):
    """For each position of `next_ids`: ("gpu", position in current_ids) to reuse
    a chunk already on the GPU, or ("cpu", chunk id) to copy it from its slot."""
    return [("gpu", current_ids.index(cid)) if cid in current_ids else ("cpu", cid)
            for cid in next_ids]


def _pin(tensor):
    if torch.cuda.is_available():
        nbytes = tensor.numel() * tensor.element_size()
        torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), nbytes, 0))


def _unpin(tensor):
    if torch.cuda.is_available():
        torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr()))


class PrefetchBank:

    def __init__(self, num_layers, num_slots, max_retrieve, device):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.max_retrieve = max_retrieve
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.pool = None  # [slots, layers, K/V, tokens, heads, dim], pinned CPU memory
        self.buffers = None  # [A/B, layers, K/V, max_retrieve, tokens, heads, dim] on device
        self.slot_of = {}  # chunk id -> slot
        self.free_slots = list(range(num_slots - 1, -1, -1))
        self.current = 0  # buffer read by the chunk being generated
        self.current_ids = []  # chunk ids in the current buffer, in order
        self.next_ids = None  # chunk ids being copied into the other buffer
        self.chunk_bytes = 0
        self.cpu_bytes_next = self.cpu_bytes_last = 0  # copied CPU -> GPU
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.ready = None

    def _allocate(self, k):
        """Pool and buffers, shaped like one layer's stored K ([1, tokens, heads, dim])."""
        entry = tuple(k.shape[1:])
        self.pool = torch.empty((self.num_slots, self.num_layers, 2) + entry, dtype=k.dtype)
        _pin(self.pool)
        self.buffers = torch.empty((2, self.num_layers, 2, self.max_retrieve) + entry,
                                   dtype=k.dtype, device=self.device)
        self.chunk_bytes = self.pool[0].numel() * self.pool.element_size()

    def store(self, chunk_id, layer_idx, k, v):
        """Copy one layer of a new chunk into its slot; returns the (k, v) slot views."""
        if self.pool is None:
            self._allocate(k)
        if layer_idx == 0:
            if not self.free_slots:
                raise RuntimeError(f"chunk {chunk_id}: no free slot ({self.num_slots} planned)")
            self.slot_of[chunk_id] = self.free_slots.pop()
        slot = self.pool[self.slot_of[chunk_id], layer_idx]
        slot[0].copy_(k[0])
        slot[1].copy_(v[0])
        return slot[0].unsqueeze(0), slot[1].unsqueeze(0)

    def free(self, chunk_ids):
        for cid in chunk_ids:
            self.free_slots.append(self.slot_of.pop(cid))

    def prefetch(self, next_ids):
        """Start filling the other buffer with `next_ids` (call before queuing this chunk)."""
        self.next_ids = list(next_ids)
        self.cpu_bytes_next = 0
        if not next_ids:
            return
        target = 1 - self.current
        if self.cuda:
            # The other buffer was read by the previous chunk: wait until it is done.
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.copy_stream) if self.cuda else nullcontext():
            for pos, (source, index) in enumerate(copy_plan(self.current_ids, next_ids)):
                if source == "gpu":
                    self.buffers[target, :, :, pos].copy_(
                        self.buffers[self.current, :, :, index], non_blocking=True)
                    continue
                slot = self.pool[self.slot_of[index]]
                for layer in range(self.num_layers):
                    for kv in (0, 1):
                        self.buffers[target, layer, kv, pos].copy_(slot[layer, kv], non_blocking=True)
                self.cpu_bytes_next += self.chunk_bytes
            if self.cuda:
                self.ready = torch.cuda.Event()
                self.ready.record(self.copy_stream)

    def use(self, chunk_ids):
        """Switch to the buffer prefetched for `chunk_ids`, waiting for its copies."""
        if list(chunk_ids) != self.next_ids:
            raise RuntimeError(f"retrieving {list(chunk_ids)}, but {self.next_ids} were prefetched")
        if self.cuda:
            torch.cuda.current_stream(self.device).wait_event(self.ready)
        self.current = 1 - self.current
        self.current_ids = self.next_ids
        self.cpu_bytes_last = self.cpu_bytes_next

    def retrieved(self, layer_idx):
        """(k, v) of the current buffer for one layer, shaped [1, chunks x tokens, heads, dim]."""
        n = len(self.current_ids)
        kv = self.buffers[self.current, layer_idx, :, :n]
        k, v = kv[0], kv[1]
        return k.reshape(1, -1, *k.shape[2:]), v.reshape(1, -1, *v.shape[2:])

    def close(self):
        """Wait for copies, then release the pinned pool and the buffers."""
        if self.cuda:
            self.copy_stream.synchronize()
        if self.pool is not None:
            _unpin(self.pool)
        self.pool = self.buffers = None
