"""Atomic run checkpoints, with one RNG snapshot per participating rank."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.distributed.fsdp import (
    FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType,
)


def quarter_steps(total):
    if total < 1:
        raise ValueError("total updates must be positive")
    return sorted({(total * fraction + 3) // 4 for fraction in (1, 2, 3, 4)})


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_resume(path):
    if not path:
        return None
    path = Path(path).resolve()
    if path.name == "latest.json":
        entry = json.loads(path.read_text())["checkpoint"]
        path = path.parent / entry
    if not (path / "manifest.json").is_file():
        raise ValueError(f"No complete Orthrus checkpoint at {path}")
    return path


def _rng_state(device):
    name, state, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [name, state.tolist(), position, has_gauss, cached],
            "accelerator": getattr(torch, device.type).get_rng_state(device).cpu()
            if device.type in {"cuda", "npu"} else None}


def restore_rng(path, rank, device):
    state = torch.load(path / f"rank-{rank:03d}.pt", map_location="cpu", weights_only=True)
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, array, pos, has, cached = state["numpy"]
    np.random.set_state((name, np.array(array, dtype=np.uint32), pos, has, cached))
    if state["accelerator"] is not None:
        getattr(torch, device.type).set_rng_state(state["accelerator"], device)


def save(root, step, model, optimizer, manifest, rank, world, device):
    from torch import distributed as dist
    distributed = world > 1
    ckpts = root / "checkpoints"
    leaf = f"checkpoint-{step:06d}"
    final = ckpts / leaf
    # Reuse a staging directory only when all ranks will overwrite their own
    # artifacts. Only the CLI's exclusive run lock can publish a checkpoint.
    pending = ckpts / ("." + leaf + ".staging")
    if rank == 0:
        ckpts.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite an existing checkpoint {final}")
        pending.mkdir(exist_ok=True)
    if distributed:
        dist.barrier()
    torch.save(_rng_state(device), pending / f"rank-{rank:03d}.pt")
    if distributed:
        # Accelerator states must fit in host memory for the rank-0 snapshot.
        # CPU states are already there; CPU-to-CPU offload can free their backing storage.
        offload = device.type != "cpu"
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=True),
        ):
            adapter_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        adapter_state = {n: t.detach().cpu() for n, t in model.state_dict().items()}
        optim_state = optimizer.state_dict()
    if rank == 0:
        torch.save(adapter_state, pending / "adapter.pt")
        torch.save({"optimizer": optim_state, "step": step}, pending / "trainer.pt")
        write_json(pending / "manifest.json", manifest | {"step": step, "format": 1})
    if distributed:
        dist.barrier()
    if rank == 0:
        os.rename(pending, final)
        write_json(ckpts / "latest.json", {"step": step, "checkpoint": leaf})
        print(f"Saved {final}", flush=True)
    if distributed:
        dist.barrier()


def restore_optimizer(model, optimizer, path, rank, world):
    if world == 1 or rank == 0:
        state = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)["optimizer"]
    else:
        state = None
    if world > 1:
        state = FSDP.scatter_full_optim_state_dict(state, model, optim=optimizer)
    optimizer.load_state_dict(state)
