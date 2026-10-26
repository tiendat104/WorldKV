"""Check that the GPUs of this machine can talk to each other.

On some rented machines (e.g. RunPod 4x A40, GPUs linked only by PCIe), the
GPUs hang forever when copying to each other directly; NCCL_P2P_DISABLE=1
(the runner's --no-p2p) fixes it. This runs the same kinds of messages the
model uses (broadcast_object_list, all_reduce) with a short timeout.

Usage, from the repo root:
  timeout 120 python -m torch.distributed.run --nproc_per_node=4 -m eval.nccl_check
"""

import datetime
import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(seconds=60))
    world = dist.get_world_size()

    obj = [42] if rank == 0 else [None]
    dist.broadcast_object_list(obj, src=0)  # the call that hung on the A40 machine
    x = torch.full((64 * 1024 * 1024,), float(rank), device="cuda" if cuda else "cpu")  # 256 MB
    dist.all_reduce(x)
    if cuda:
        torch.cuda.synchronize()
    expected = world * (world - 1) / 2
    ok = obj[0] == 42 and x[0].item() == expected
    print(f"rank {rank}: {'OK' if ok else 'WRONG RESULT'} "
          f"(sum {x[0].item():.0f}, expected {expected:.0f})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
