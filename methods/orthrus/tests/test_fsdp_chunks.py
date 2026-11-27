"""Count FSDP adapter all-gathers across gradient-accumulating chunks."""
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


class ChunkCollectiveTests(unittest.TestCase):
    def test_one_gather_per_update_with_shard_grad_op(self):
        # FSDP schedules real unshard/reshard calls; the transport simulates
        # seven ranks on one CPU process. No network or accelerator is needed.
        try:
            from torch.distributed.fsdp._flat_param import FlatParamHandle
            import torch.testing._internal.distributed.fake_pg  # noqa: F401
        except ImportError:
            self.skipTest("Torch's simulated process group is unavailable")
        if dist.is_initialized():
            self.skipTest("This isolated test creates its own process group")
        dist.init_process_group("fake", rank=0, world_size=7, store=dist.HashStore())
        self.addCleanup(dist.destroy_process_group)

        for strategy, expected in ((ShardingStrategy.FULL_SHARD, [4, 4]),
                                   (ShardingStrategy.SHARD_GRAD_OP, [1, 1])):
            with self.subTest(strategy=strategy.name):
                model = FSDP(torch.nn.Linear(16, 16), device_id=torch.device("cpu"),
                             use_orig_params=True, sharding_strategy=strategy)
                opt = torch.optim.AdamW(model.parameters())
                gather = FlatParamHandle._all_gather_flat_param
                per_update = []

                def count_gather(handle, *args, **kwargs):
                    per_update[-1] += 1
                    return gather(handle, *args, **kwargs)

                with patch.object(FlatParamHandle, "_all_gather_flat_param", count_gather):
                    for _ in range(2):
                        per_update.append(0)
                        opt.zero_grad(set_to_none=True)
                        for chunk in range(4):
                            with model.no_sync() if chunk < 3 else nullcontext():
                                model(torch.ones(1, 16)).sum().backward()
                        opt.step()
                self.assertEqual(per_update, expected)
