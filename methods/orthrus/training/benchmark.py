"""Measure full-size training updates using the epoch's actual batches and optimizer."""
import hashlib
import json
import time

import torch
from torch import distributed as dist


def _plan_digests(plan, device, world):
    serialized = json.dumps(plan, separators=(",", ":")).encode()
    digest = hashlib.sha256(serialized).digest()
    if world == 1:
        return [digest.hex()]
    # Gather outside the timed window. Use a device tensor on HCCL / NCCL.
    local = torch.tensor(list(digest), dtype=torch.int32, device=device)
    values = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(values, local)
    return [bytes(tensor.cpu().tolist()).hex() for tensor in values]


def summarize(steps, elapsed_seconds):
    count = len(steps)
    if not count or elapsed_seconds <= 0:
        raise ValueError("Expected measured training updates and positive elapsed time")
    documents = sum(s["documents"] for s in steps)
    tokens = sum(s["context_tokens"] for s in steps)
    return {
        "measured_updates": count, "wall_seconds": elapsed_seconds,
        "seconds_per_update": elapsed_seconds / count,
        "context_tokens_per_second": tokens / elapsed_seconds, "documents_per_second": documents / elapsed_seconds,
        "context_tokens": tokens, "documents": documents, "anchors": sum(s["anchors"] for s in steps),
        "mean_max_teacher_seconds": sum(s["max_teacher_seconds"] for s in steps) / count,
        "teacher_gib_per_update": sum(s["teacher_bytes"] for s in steps) / (count * 2**30),
    }


def run_benchmark(loader, plan, update, steps, warmup, device, rank, world):
    """Time the maximum rank wall time. Include loader, prefill, training and communication."""
    digests = _plan_digests(plan[:steps], device, world)
    if rank == 0:
        print(f"Packed training benchmark: {warmup} warmup + {steps - warmup} measured updates", flush=True)
    iterator = iter(loader)
    for step in range(1, warmup + 1):
        update(step, next(iterator))
        if rank == 0:
            print(f"Warmup {step}/{warmup}", flush=True)
    if rank == 0:
        print(f"Measuring updates {warmup + 1} through {steps} ...", flush=True)
    accelerator = getattr(torch, device.type) if device.type != "cpu" else None
    memory_stats = accelerator is not None and all(hasattr(accelerator, name) for name in (
        "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"))
    if accelerator is not None:
        accelerator.synchronize(device)
    if world > 1:
        dist.barrier()
    if accelerator is not None:
        accelerator.synchronize(device)
        if memory_stats:
            accelerator.reset_peak_memory_stats(device)
    start = time.perf_counter()
    recorded = []
    for step in range(warmup + 1, steps + 1):
        metrics, lr = update(step, next(iterator))
        if rank == 0:
            recorded.append({"step": step, "lr": lr, **metrics})
            print(f"Measured update {step - warmup}/{steps - warmup}, context tokens {metrics['context_tokens']:,}",
                  flush=True)
    if accelerator is not None:
        accelerator.synchronize(device)
    elapsed = time.perf_counter() - start
    if world > 1:
        duration = torch.tensor(elapsed, device=device, dtype=torch.float32)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        elapsed = duration.item()
    memory = {}
    if memory_stats:
        peaks = torch.tensor([accelerator.max_memory_allocated(device), accelerator.max_memory_reserved(device)],
                             device=device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        allocated, reserved = peaks.cpu().tolist()
        memory = {"max_rank_peak_allocated_gib": allocated / 2**30, "max_rank_peak_reserved_gib": reserved / 2**30}
    if rank == 0:
        return summarize(recorded, elapsed) | memory | {
            "warmup_updates": warmup, "total_benchmark_updates": steps,
            "first_measured_step": warmup + 1, "last_measured_step": steps,
            "rank_batch_plan_sha256": digests, "updates": recorded,
        }
    return None
