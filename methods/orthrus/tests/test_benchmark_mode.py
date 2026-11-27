import unittest
from unittest.mock import patch

import torch

from training.benchmark import run_benchmark


class FullBatchBenchmarkTests(unittest.TestCase):
    def test_warmup_updates_run_without_entering_window(self):
        plan = [[7, 8], [9], [20, 21, 22], [42], [99]]
        seen = []

        def update(step, rows):
            seen.append((step, rows))
            return {"documents": len(rows), "context_tokens": sum(rows), "anchors": step,
                    "max_teacher_seconds": step / 4, "teacher_bytes": step * 2**30}, 0.001 * step

        with patch("training.benchmark.time.perf_counter", side_effect=[10, 14]):
            result = run_benchmark(plan, plan, update, steps=4, warmup=2, device=torch.device("cpu"),
                                   rank=0, world=1)
        self.assertEqual(seen, list(enumerate(plan[:4], 1)))
        self.assertEqual([s["step"] for s in result["updates"]], [3, 4])
        self.assertEqual(result["seconds_per_update"], 2)
        self.assertEqual(result["context_tokens_per_second"], 105 / 4)
        self.assertEqual(result["teacher_gib_per_update"], 3.5)
        self.assertEqual(result["mean_max_teacher_seconds"], 0.875)
        self.assertEqual(result["documents"], 4)
        self.assertEqual(len(result["rank_batch_plan_sha256"][0]), 64)


if __name__ == "__main__":
    unittest.main()
