"""Generate many videos with one loaded model.

generate_fast.py loads LingBot (~81 GB of files) for every video, which takes
minutes each time. This worker loads it once, then generates every job in a
jobs file one after another. Each job holds the command-line arguments that
generate_fast.py would take; they are parsed by generate_fast.py's own parser
and passed to the same generate() call, so the videos are the same as with
one generate_fast.py run per video.

eval.runner starts it. By hand, from the repo root:
  python -m torch.distributed.run --nproc_per_node=4 -m eval.worker JOBS.json

The jobs file is a JSON list of
  {"name", "argv", "save_file", "telemetry", "meta", "meta_fields"}.
Before and after each job, the worker writes the job's "meta" file with its
status ("running" / "done") and timings. If the worker crashes, the job still
marked "running" is the one that failed, and eval.runner restarts the worker
for the remaining jobs.
"""

import gc
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from PIL import Image

import generate_fast as gf
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video

from eval.telemetry import ENV_VAR

# Settings that belong to the loaded model, so they must match across jobs.
MODEL_ARGS = ("task", "ckpt_dir", "t5_fsdp", "dit_fsdp", "ulysses_size", "t5_cpu",
              "convert_model_dtype")


def parse_job_args(argv):
    """generate_fast.py's own argument parsing and checks, for one job."""
    saved = sys.argv
    sys.argv = ["generate_fast.py"] + list(argv)
    try:
        return gf._parse_args()
    finally:
        sys.argv = saved


def write_meta(path, meta):
    if int(os.getenv("RANK", 0)) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=1)


def main():
    with open(sys.argv[1]) as f:
        jobs = json.load(f)
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    gf._init_logging(rank)

    all_args = [parse_job_args(job["argv"]) for job in jobs]
    first = all_args[0]
    for job, args in zip(jobs, all_args):
        for key in MODEL_ARGS:
            if getattr(args, key) != getattr(first, key):
                raise ValueError(f"{job['name']}: --{key} differs from the first job, "
                                 "but one worker loads one model.")

    # Same distributed setup as generate_fast.generate().
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    if first.ulysses_size > 1:
        assert first.ulysses_size == world_size, "ulysses_size must equal the number of GPUs."
        init_distributed_group()

    cfg = WAN_CONFIGS[first.task]
    start = time.time()
    pipeline = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=first.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=first.t5_fsdp,
        dit_fsdp=first.dit_fsdp,
        use_sp=(first.ulysses_size > 1),
        t5_cpu=first.t5_cpu,
        convert_model_dtype=first.convert_model_dtype,
    )
    load_time = time.time() - start
    logging.info(f"Model loaded once in {load_time:.0f} s; {len(jobs)} jobs to run.")

    for job, args in zip(jobs, all_args):
        if os.path.exists(job["save_file"]):
            logging.info(f"[skip] {job['name']}: video exists")
            continue
        meta = dict(job.get("meta_fields", {}), status="running", argv=job["argv"],
                    model_load_s=round(load_time, 1))
        write_meta(job["meta"], meta)
        logging.info(f"[job] {job['name']}")
        os.environ[ENV_VAR] = job["telemetry"]  # read by eval/telemetry.py inside generate()
        job_start = time.time()

        offload_model = args.offload_model
        if offload_model is None:  # generate_fast.py's default
            offload_model = world_size <= 1
        # The same call as in generate_fast.generate().
        video = pipeline.generate(
            args.prompt,
            Image.open(args.image).convert("RGB"),
            action_path=args.action_path,
            chunk_size=3,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            seed=args.base_seed,
            offload_model=offload_model,
            max_attention_size=args.max_attention_size,
            use_retrieval=args.use_retrieval,
            retrieval_frames=args.retrieval_frames,
            retrieval_rope_correction=args.retrieval_rope_correction,
            full_kv=args.full_kv,
            sliding_window=args.sliding_window,
            kv_compression_enable=args.kv_compression_enable,
            kv_compression_keep_ratio=args.kv_compression_keep_ratio,
            kv_compression_anchor_rotate=args.kv_compression_anchor_rotate,
            kv_compression_at_store=args.kv_compression_at_store,
            kv_compression_pooled=args.kv_compression_pooled,
            kv_bank_on_gpu=args.kv_bank_on_gpu)

        if rank == 0:
            os.makedirs(os.path.dirname(job["save_file"]), exist_ok=True)
            save_video(tensor=video[None], save_file=job["save_file"], fps=cfg.sample_fps,
                       nrow=1, normalize=True, value_range=(-1, 1))
        # Free this video's caches before the next job.
        del video
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        meta.update(status="done", wall_time_s=round(time.time() - job_start, 1))
        write_meta(job["meta"], meta)

    logging.info("Worker finished.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
