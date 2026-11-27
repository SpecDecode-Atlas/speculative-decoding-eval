"""One-epoch Orthrus training on pretokenized Qwen3 PerfectBlend rollouts."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.utils.data import DataLoader

from . import checkpoint
from .data import (Documents, append_tail, identity_collate, lengths_index, load_records,
                   local_plan, sample_blocks, slot_weights, tensor_rows)
from .objective import hidden_gradient


def learning_rate(step, total, warmup=50, peak=6e-4):
    """LR for the 1-based optimizer update; the first update is nonzero."""
    if not 1 <= step <= total:
        raise ValueError("Step outside optimizer schedule")
    if step <= warmup:
        return peak * step / warmup
    return peak * 0.5 * (1.0 + math.cos(math.pi * (step - warmup - 1) / (total - warmup)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "dataset", "output-dir"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--teacher-socket", help="Dedicated AR server; omit for local frozen teacher prefills")
    p.add_argument("--fsdp-strategy", choices=("shard-grad-op", "full-shard"), default="shard-grad-op",
                   help="Keep gathered adapter parameters across unsynchronized chunks or release after each backward")
    p.add_argument("--device", choices=["npu", "cuda", "cpu"], default="npu")
    p.add_argument("--doc-max", type=int, default=3072)
    p.add_argument("--pack-tokens", type=int, default=9216)
    p.add_argument("--max-anchors", type=int, default=768)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--mask-token-id", type=int, default=151669)
    p.add_argument("--blocks-per-chunk", type=int, default=64)
    p.add_argument("--head-chunk-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="One real update on up to two blocks of a short document per rank")
    p.add_argument("--benchmark-steps", type=int, metavar="N",
                   help="Run the first N full-size packed updates using the epoch schedule; write a timing report")
    p.add_argument("--benchmark-warmup", type=int, default=3, metavar="N",
                   help="Exclude this many initial updates from packed benchmark timing (default: 3)")
    args = p.parse_args()
    if (args.doc_max < 2 or args.pack_tokens < args.doc_max or args.block_size < 2 or
            min(args.head_chunk_size, args.blocks_per_chunk, args.log_every, args.max_anchors) < 1 or args.workers < 0):
        p.error("Invalid dimensions, token budget, or worker count")
    if args.smoke and args.resume:
        p.error("Smoke runs use a fresh output")
    if args.benchmark_steps is not None:
        if args.smoke or args.resume:
            p.error("Use a fresh full-size benchmark without --smoke or --resume")
        if not 0 <= args.benchmark_warmup < args.benchmark_steps:
            p.error("Require 0 <= --benchmark-warmup < --benchmark-steps")
    return args


def reduce(value, device, world, op=dist.ReduceOp.SUM, dtype=torch.float32):
    t = torch.as_tensor(value, device=device, dtype=dtype)
    if world > 1:
        dist.all_reduce(t, op=op)
    return t


def init_device(kind, world, local_rank):
    if kind == "npu":
        import torch_npu  # noqa: F401 -- registers the NPU and HCCL backends
    if kind != "cpu":
        getattr(torch, kind).set_device(local_rank)
    device = torch.device(kind, local_rank) if kind != "cpu" else torch.device("cpu")
    if world > 1:
        backend = {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[kind]
        dist.init_process_group(backend, timeout=timedelta(hours=3))
    return device


def plan_epoch(args, data, world, rank, device):
    cache = args.cache_dir or args.output_dir.resolve().parent / ".orthrus-data-cache"
    # The check-only stage never scans the full 1M-row dataset.
    if args.smoke:
        data = data.select(range(min(len(data), max(256, world * 16))))
    rows, lengths = lengths_index(data, args.dataset, args.doc_max, cache)
    if not len(rows):
        raise ValueError("No trainable rows in the selected data after truncation")
    if args.smoke:
        plan = [[int(rows[rank % len(rows)])]]
    else:
        plan = local_plan(rows, lengths, args.pack_tokens, world, rank, args.seed)
        used = int(reduce(sum(map(len, plan)), device, world, dtype=torch.int64).item())
        append_tail(plan, rows, used, world, rank, args.seed)
    return Documents(data, args.doc_max, smoke=args.smoke), plan, len(rows), len(data) - len(rows)


def model_identity(path):
    # A user can move a run directory; teacher identity includes config and the
    # resolved snapshot path. For immutable HF snapshots the directory is its commit.
    import hashlib
    return {"model": str(path.resolve()), "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest()}


def dummy_features(target, device):
    c = target.config
    h = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    kv = torch.zeros(1, c.num_key_value_heads, 1, h, device=device, dtype=target.lm_head.weight.dtype)
    return tuple((kv, kv) for _ in range(c.num_hidden_layers))


def context_for(row, target, client, device):
    from orthrus_atlas import cache_layers
    ids = row["ids"]
    if int(ids.min()) < 0 or int(ids.max()) >= target.config.vocab_size:
        raise ValueError(f"Out-of-vocabulary token in dataset row {row['row']}")
    if client is not None:
        return client.infer(ids, device)
    with torch.no_grad():
        out = target.model(input_ids=ids.to(device)[None, :], use_cache=True, return_dict=True)
    return cache_layers(out.past_key_values), out.last_hidden_state


def train_step(model, target, optimizer, client, rows, args, world, rank, step, device):
    blocks = sample_blocks(rows, min(2, args.max_anchors) if args.smoke else args.max_anchors,
                           np.random.SeedSequence([args.seed, step, rank]))
    parts = []
    denom = 0.0
    n = args.block_size - 1
    for doc, (row, anchors) in enumerate(zip(rows, blocks)):
        if not len(anchors):
            continue
        for chunk in anchors.split(args.blocks_per_chunk):
            weights = slot_weights(row, chunk, n)
            parts.append((doc, chunk, weights))
            denom += float(weights.sum())
    global_weight = float(reduce(denom, device, world).item())
    count = int(reduce(len(parts), device, world, dist.ReduceOp.MAX, torch.long).item())
    if not global_weight or count == 0:
        raise ValueError("Packed step has no supervised positions")
    optimizer.zero_grad(set_to_none=True)
    stats = torch.zeros(2, device=device)
    active_doc, kv, teacher = None, None, None
    # Construct repeated target positions on the host to avoid a device-side
    # dynamic shape synchronization when deduplicating their vocabulary rows.
    offsets = torch.arange(n)
    start_wait = client.wait_seconds if client else 0.0
    start_bytes = client.received_bytes if client else 0
    for slot in range(count):
        # Every rank runs equally many forward/backward collectives, even the final ragged batch.
        sync = model.no_sync() if world > 1 and slot + 1 < count else nullcontext()
        with sync:
            if slot < len(parts):
                doc, cpu_anchors, weights = parts[slot]
                if doc != active_doc:
                    kv, teacher = None, None
                    kv, teacher = context_for(rows[doc], target, client, device)
                    active_doc = doc
                anchor_ids = rows[doc]["ids"][cpu_anchors].to(device)
                anchors = cpu_anchors.to(device)
                hidden = model(anchor_ids, anchors, kv, max_context_length=int(cpu_anchors.max()))
                positions = (cpu_anchors[:, None] + offsets).clamp_max(teacher.shape[1] - 1)
                gradient, sums = hidden_gradient(hidden, teacher[0], weights.to(device), target.lm_head,
                                                 args.head_chunk_size, world / global_weight,
                                                 teacher_positions=positions)
                hidden.backward(gradient)
                stats += sums
            else:
                # A short bin makes no contribution, but participates in the same FSDP reduction.
                zero = torch.zeros(1, device=device, dtype=torch.long)
                hidden = model(zero, zero, dummy_features(target, device))
                hidden.backward(torch.zeros_like(hidden))
    clip = model.clip_grad_norm_(1.0) if world > 1 else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not math.isfinite(float(clip)):
        raise FloatingPointError(f"Nonfinite gradient norm on rank {rank}, update {step}")
    optimizer.step()
    if world > 1:
        dist.all_reduce(stats)
    ce, tv = (stats / global_weight).cpu().tolist()
    if not math.isfinite(ce + tv):
        raise FloatingPointError(f"Nonfinite loss at update {step}")
    docs, tokens, anchors, nbytes = reduce(
        [len(rows), sum(len(row["ids"]) for row in rows), sum(map(len, blocks)),
         client.received_bytes - start_bytes if client else 0],
        device, world, dtype=torch.int64).cpu().tolist()
    wait = float(reduce(client.wait_seconds - start_wait if client else 0., device, world,
                        dist.ReduceOp.MAX).item())
    return {"loss": 0.1 * ce + 0.9 * tv, "ce": ce, "tv": tv, "grad_norm": float(clip), "weight": global_weight,
            "documents": docs, "context_tokens": tokens, "anchors": anchors,
            "max_teacher_seconds": wait, "teacher_bytes": nbytes}


def main():
    from transformers import AutoModelForCausalLM
    from orthrus_atlas import OrthrusView
    args = parse_args()
    rank, local_rank, world = [int(os.environ.get(k, default)) for k, default in
                               (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1))]
    device = init_device(args.device, world, local_rank)
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=True, parents=True)
    resume = checkpoint.read_resume(args.resume)
    if rank == 0:
        run_lock = (output / ".train.lock").open("a")
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.benchmark_steps and any((output / filename).exists() for filename in (
                "run.json", "metrics.jsonl", "progress.json", "checkpoints", "benchmark.json")):
            raise ValueError("Choose a fresh benchmark output directory. Existing training artifacts are present.")
        if (output / "run.json").exists() and resume is None:
            raise ValueError("A training run already uses this directory. Supply --resume or choose a fresh run.")
        if resume and (output / "checkpoints/latest.json").exists():
            latest = checkpoint.read_resume(output / "checkpoints/latest.json")
            if resume != latest:
                raise ValueError("Resume an earlier checkpoint into a new output directory.")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    data = load_records(args.dataset)
    if rank == 0:
        print("Reading rollout lengths and building the native one-epoch Multipack plan...", flush=True)
    records, plan, eligible, skipped = plan_epoch(args, data, world, rank, device)
    total = len(plan)
    if not total:
        raise RuntimeError("Empty optimizer plan")
    if args.benchmark_steps and args.benchmark_steps > total:
        raise ValueError(f"Benchmark requests {args.benchmark_steps} updates but this epoch has {total}")
    identity = {**model_identity(args.model), "dataset": str(args.dataset.resolve()), "dataset_fingerprint": data._fingerprint,
                "seed": args.seed, "world_size": world, "optimizer_steps": total, "document_max_length": args.doc_max,
                "pack_tokens": args.pack_tokens, "max_anchors": args.max_anchors, "block_size": args.block_size,
                "mask_token_id": args.mask_token_id, "blocks_per_chunk": args.blocks_per_chunk,
                "head_chunk_size": args.head_chunk_size, "smoke": args.smoke, "method": "orthrus",
                "teacher_mode": "dedicated" if args.teacher_socket else "local",
                "peak_lr": 6e-4, "warmup": 50, "weight_decay": 0.01, "betas": [0.9, 0.999],
                "eps": 1e-8, "grad_clip": 1., "loss": {"ce": 0.1, "tv": 0.9}, "loss_decay": 7.,
                "epochs": 1, "train_data_ratio": 1.0, "optimizer": "AdamW", "num_proposals": args.block_size - 1,
                "schedule": "linear warmup then cosine", "gradient_accumulation": 1,
                "weights_dtype": "float32", "compute_dtype": "float32" if device.type == "cpu" else "bfloat16",
                "distributed_strategy": f"FSDP {args.fsdp_strategy.upper().replace('-', '_')}" if world > 1 else "single rank",
                "gradient_checkpointing": True, "attention": "SDPA", "dropout": 0.0,
                "num_workers": args.workers, "eligible_rows": eligible, "excluded_rows": skipped}
    initial_step, elapsed_base, docs_base, tokens_base = 0, 0., 0, 0
    if resume:
        saved = json.loads((resume / "manifest.json").read_text())
        # These change the placement of equivalent teacher calculations and
        # FSDP buffers. The checkpoint stores a full adapter and Adam state.
        execution_keys = {"teacher_mode", "distributed_strategy"}
        changed = sorted(key for key in set(identity) | set(saved["training"])
                         if saved["training"].get(key) != identity.get(key))
        if set(changed) - execution_keys:
            raise ValueError("Resume changes the training protocol: " + ", ".join(sorted(set(changed) - execution_keys)))
        if changed and rank == 0:
            print("Resuming with updated execution settings: " + ", ".join(
                f"{key}={saved['training'].get(key)} -> {identity.get(key)}" for key in changed), flush=True)
        initial_step, elapsed_base = saved["step"], saved["elapsed_seconds"]
        docs_base, tokens_base = saved["documents_seen"], saved["context_tokens_seen"]
    if rank == 0:
        saves = [] if args.benchmark_steps else checkpoint.quarter_steps(total)
        config = identity | {"save_updates": saves}
        if args.benchmark_steps:
            config.update(benchmark_steps=args.benchmark_steps, benchmark_warmup=args.benchmark_warmup)
        checkpoint.write_json(output / "run.json", config)
        print(f"Eligible rows {eligible:,}, excluded {skipped:,}, updates {total:,}, save at {saves}", flush=True)
    if initial_step == total:
        if rank == 0:
            print("The requested checkpoint completed the epoch.", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True, dtype=dtype,
                                                 attn_implementation="sdpa").eval().requires_grad_(False).to(device)
    view = OrthrusView(target, mask_token_id=args.mask_token_id, block_size=args.block_size, gradient_checkpointing=True)
    if resume:
        view.load_state_dict(torch.load(resume / "adapter.pt", mmap=True, map_location="cpu", weights_only=True))
    strategy = ShardingStrategy.SHARD_GRAD_OP if args.fsdp_strategy == "shard-grad-op" else ShardingStrategy.FULL_SHARD
    model = FSDP(view, device_id=device, sharding_strategy=strategy, use_orig_params=True,
                 limit_all_gathers=True) if world > 1 else view
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01, foreach=False)
    if resume:
        checkpoint.restore_optimizer(model, optimizer, resume, rank, world)
        checkpoint.restore_rng(resume, rank, device)

    loader = DataLoader(records, batch_sampler=plan[initial_step:args.benchmark_steps], num_workers=args.workers,
                        collate_fn=identity_collate, generator=torch.Generator().manual_seed(args.seed + rank),
                        **({"multiprocessing_context": "spawn"} if args.workers else {}))
    client = None
    if args.teacher_socket:
        from teacher.client import TeacherClient
        client = TeacherClient(args.teacher_socket, model_path=args.model)

    def update(step, rows):
        lr = learning_rate(step, total)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return train_step(model, target, optimizer, client, tensor_rows(rows), args, world, rank, step, device), lr

    if args.benchmark_steps:
        from .benchmark import run_benchmark
        report = run_benchmark(loader, plan, update, args.benchmark_steps, args.benchmark_warmup, device, rank, world)
        if rank == 0:
            path = output / "benchmark.json"
            checkpoint.write_json(path, {"training": identity, **report})
            print(f"{report['seconds_per_update']:.3f} s/update, {report['context_tokens_per_second']:.1f} "
                  f"context tokens/s, {report['teacher_gib_per_update']:.3f} teacher GiB/update. Report: {path}", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    writer, log = None, None
    if rank == 0:
        logpath = output / "metrics.jsonl"
        if resume and logpath.exists():
            # Discard metrics newer than the last completed checkpoint after interruption.
            tmp = output / ".metrics.recovering"
            with logpath.open() as old, tmp.open("w") as new:
                for line in old:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item["step"] <= initial_step:
                        new.write(line)
            os.replace(tmp, logpath)
        log = logpath.open("a", buffering=1)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output / "tensorboard"), purge_step=initial_step + 1 if resume else None)
    start = prev = time.monotonic()
    saves = set(checkpoint.quarter_steps(total))
    docs_seen, tokens_seen = docs_base, tokens_base
    session_docs, session_tokens = 0, 0
    for step, rows in enumerate(loader, initial_step + 1):
        result, lr = update(step, rows)
        docs_seen += result["documents"]
        tokens_seen += result["context_tokens"]
        session_docs += result["documents"]
        session_tokens += result["context_tokens"]
        now = time.monotonic()
        elapsed = elapsed_base + now - start
        if rank == 0 and (step == initial_step + 1 or step % args.log_every == 0 or step in saves):
            duration = max(now - start, 1e-9)
            record = result | {"step": step, "total_steps": total, "fraction": step / total, "lr": lr,
                               "elapsed_seconds": elapsed, "eta_seconds": duration / (step - initial_step) * (total - step),
                               "step_seconds": now - prev, "documents_seen": docs_seen, "context_tokens_seen": tokens_seen,
                               "documents_per_second": session_docs / duration, "context_tokens_per_second": session_tokens / duration}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            checkpoint.write_json(output / "progress.json", record)
            print(f"[{step:,}/{total:,} | {100*step/total:.2f}%] loss={record['loss']:.5f} CE={record['ce']:.5f} "
                  f"TV={record['tv']:.5f} LR={lr:.3g} tokens/s={record['context_tokens_per_second']:.0f} "
                  f"elapsed={timedelta(seconds=int(elapsed))} ETA={timedelta(seconds=int(record['eta_seconds']))}", flush=True)
            if writer:
                for key, value in record.items():
                    writer.add_scalar("train/" + key, value, step)
                writer.flush()
        if step in saves:
            manifest = {"training": identity, "model": str(args.model.resolve()), "block_size": args.block_size,
                        "mask_token_id": args.mask_token_id, "elapsed_seconds": elapsed, "documents_seen": docs_seen,
                        "context_tokens_seen": tokens_seen, "world_size": world}
            checkpoint.save(output, step, model, optimizer, manifest, rank, world, device)
        prev = now
    if not args.smoke and docs_seen != eligible:
        raise RuntimeError(f"One-epoch coverage mismatch: saw {docs_seen} documents, expected {eligible}")
    if rank == 0:
        if writer:
            writer.close()
        log.close()
        print(f"{'Smoke check' if args.smoke else 'One epoch'} complete. Output: {output}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
