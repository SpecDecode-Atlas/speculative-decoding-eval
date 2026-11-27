"""Moving the causal prefill onto a trainer must preserve its optimizer update."""

import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoModelForCausalLM

from orthrus_atlas import OrthrusView
from teacher.service import PrefillServer
from test_orthrus_view import tiny_target
from training.train import train_step


class LocalTeacherTests(unittest.TestCase):
    def test_local_and_dedicated_bfloat16_prefills_produce_the_same_update(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        target = tiny_target().to(torch.bfloat16)
        args = SimpleNamespace(max_anchors=9, blocks_per_chunk=2, head_chunk_size=5,
                               block_size=4, seed=42, smoke=False)
        rows = [
            {"row": 0, "ids": torch.tensor([3, 4, 5, 6, 7, 8, 9]),
             "weights": torch.tensor([0., 0., 1., 0.5, 1., 0., 1.])},
            {"row": 1, "ids": torch.tensor([10, 7, 5, 19]),
             "weights": torch.tensor([0., 1., 1., 1.])},
        ]
        with tempfile.TemporaryDirectory() as snapshot:
            target.save_pretrained(snapshot)
            # Both production processes construct their own model from the same
            # snapshot. This also reconstructs nonpersistent RoPE buffers.
            target = AutoModelForCausalLM.from_pretrained(
                snapshot, dtype=torch.bfloat16, attn_implementation="sdpa",
            ).eval().requires_grad_(False)
            server = PrefillServer(SimpleNamespace(model=snapshot, max_length=16, log_every=10,
                                                  device="cpu", dtype="bfloat16"))

            class DedicatedPrefill:
                # The transport round-trip is tested separately on a socketpair.
                wait_seconds = 0.0
                received_bytes = 0

                def infer(self, ids, device):
                    kv, hidden, _ = server._prefill(ids.tolist())
                    tensors = [t for pair in kv for t in pair] + [hidden]
                    self.received_bytes += sum(t.numel() * t.element_size() for t in tensors)
                    return tuple((k.clone(), v.clone()) for k, v in kv), hidden.clone()

            def update(client):
                view = OrthrusView(target, mask_token_id=127, block_size=4, gradient_checkpointing=True)
                optimizer = torch.optim.AdamW(view.parameters(), lr=6e-4, weight_decay=0.01, foreach=False)
                metrics = train_step(view, target, optimizer, client, rows, args, 1, 0, 1, torch.device("cpu"))
                return metrics, view.state_dict(), optimizer.state_dict()

            remote, remote_model, remote_optim = update(DedicatedPrefill())
            local, local_model, local_optim = update(None)
            self.assertGreater(remote["teacher_bytes"], 0)
            self.assertEqual(local["teacher_bytes"], 0)
            for name in ("loss", "ce", "tv", "grad_norm", "anchors", "weight"):
                self.assertEqual(remote[name], local[name], name)
            torch.testing.assert_close(local_model, remote_model, atol=0, rtol=0)
            torch.testing.assert_close(local_optim["state"], remote_optim["state"], atol=0, rtol=0)
