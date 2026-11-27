import json
import tempfile
import unittest
from pathlib import Path

from training.compare_timing import compare


class CompareTimingTest(unittest.TestCase):
    def test_excludes_first_ten_steps_and_rejects_changed_workload(self):
        controls = dict(model="Qwen3", config_sha256="abc", dataset_fingerprint="test", world_size=7,
                        seed=42, optimizer_steps=100, document_max_length=3072, pack_tokens=9216,
                        max_anchors=768, block_size=16, mask_token_id=151669, blocks_per_chunk=64,
                        head_chunk_size=256, loss={"ce": 0.1, "tv": 0.9})
        benchmark = dict(training=controls, first_measured_step=11, last_measured_step=20,
                         context_tokens=500, documents=14, seconds_per_update=8.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps(controls))
            records = [dict(step=10, elapsed_seconds=140, context_tokens_seen=400, documents_seen=12),
                       dict(step=20, elapsed_seconds=240, context_tokens_seen=900, documents_seen=26)]
            (root / "metrics.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))
            self.assertAlmostEqual(compare(root, benchmark)["baseline_seconds_per_update"], 10)
            with self.assertRaisesRegex(ValueError, "context_tokens"):
                compare(root, dict(benchmark, context_tokens=60))
            with self.assertRaisesRegex(ValueError, "seed"):
                compare(root, dict(benchmark, training=dict(controls, seed=999)))
