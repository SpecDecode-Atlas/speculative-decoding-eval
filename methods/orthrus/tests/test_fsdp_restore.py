"""Resume a multi-document Orthrus update across FSDP sharding strategies."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


def run_distributed_check():
    import copy
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from datetime import timedelta
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp._debug_utils import SimpleProfiler
    from torch.testing._internal.distributed.multi_threaded_pg import (
        ProcessLocalGroup, _install_threaded_pg, _uninstall_threaded_pg,
    )

    from orthrus_atlas import OrthrusView
    from training import checkpoint
    from training.train import train_step
    from test_orthrus_view import tiny_target

    torch.set_num_threads(1)
    torch.manual_seed(99)
    target_template = tiny_target()
    device = torch.device("cpu")
    args = SimpleNamespace(max_anchors=12, blocks_per_chunk=2, head_chunk_size=5,
                           block_size=4, seed=42, smoke=False)
    # The second rank needs fewer chunks and must participate with dummy
    # forwards/backwards at the correct slots.
    documents = ([3, 7, 11, 15, 23, 47, 88, 1, 6, 7, 2, 9, 19], [6, 3, 55, 99, 7, 7])
    store = dist.HashStore()
    states = {}
    norms = {}

    def worker(rank, root):
        try:
            dist.init_process_group("threaded", rank=rank, world_size=2, store=store, timeout=timedelta(seconds=15))
            ids = torch.tensor(documents[rank])
            rows = [{"row": rank, "ids": ids, "weights": (torch.arange(len(ids)) >= 2).float()}]
            for strategy in (ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP):
                target = copy.deepcopy(target_template)
                view = OrthrusView(target, mask_token_id=127, block_size=4, gradient_checkpointing=True)
                path = root / "checkpoints/checkpoint-000001"
                if strategy == ShardingStrategy.SHARD_GRAD_OP:
                    view.load_state_dict(torch.load(path / "adapter.pt", map_location="cpu", weights_only=True))
                model = FSDP(view, device_id=device, use_orig_params=True, sharding_strategy=strategy)
                opt = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
                if strategy == ShardingStrategy.FULL_SHARD:
                    train_step(model, target, opt, None, rows, args, 2, rank, 1, device)
                    checkpoint.save(root, 1, model, opt, {}, rank, 2, device)
                else:
                    checkpoint.restore_optimizer(model, opt, path, rank, 2)
                result = train_step(model, target, opt, None, rows, args, 2, rank, 2, device)
                norms[strategy, rank] = result["grad_norm"]
                with FSDP.summon_full_params(model, writeback=False):
                    states[strategy, rank] = {k: p.detach().clone() for k, p in model.module.named_parameters()}
                dist.barrier()
            dist.destroy_process_group()
        except BaseException as exc:
            ProcessLocalGroup.exception_handle(exc)
            raise

    torch._C._distributed_c10d._set_thread_isolation_mode(True)
    _install_threaded_pg()
    try:
        # The FSDP internal profiler is process-global, while these two
        # transport threads emulate independent rank processes.
        with tempfile.TemporaryDirectory() as tmp, patch.object(SimpleProfiler, "profile", side_effect=lambda *_: nullcontext()):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, rank, Path(tmp)) for rank in range(2)]
                for future in futures:
                    future.result()
    finally:
        _uninstall_threaded_pg()
        torch._C._distributed_c10d._set_thread_isolation_mode(False)
    expected = states[ShardingStrategy.FULL_SHARD, 0]
    for key, result in states.items():
        assert norms[key] == norms[ShardingStrategy.FULL_SHARD, 0]
        for name, tensor in result.items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


class FSDPRestoreTests(unittest.TestCase):
    def test_next_orthrus_update_matches_after_strategy_change(self):
        try:
            import torch.testing._internal.distributed.multi_threaded_pg  # noqa: F401
        except ImportError:
            self.skipTest("Torch's threaded process group is unavailable")
        method = Path(__file__).resolve().parents[1]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PYTHONPATH": os.pathsep.join([str(method), os.environ.get("PYTHONPATH", "")])}
        # Isolate the process-group installation and enforce a real deadline
        # if either logical rank misses a collective.
        result = subprocess.run([sys.executable, __file__], env=env, cwd=method,
                                timeout=30, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    run_distributed_check()
