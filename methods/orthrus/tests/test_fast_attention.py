"""Reference equivalence for dropping unreachable clean history before attention."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from orthrus_atlas import OrthrusView, cache_layers


class TrimmedHistoryTests(unittest.TestCase):
    def test_same_outputs_and_gradients_with_shorter_sdpa_key_axis(self):
        torch.manual_seed(321)
        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        doc = torch.randint(0, 64, (1, 80))
        with torch.no_grad():
            kv = cache_layers(target.model(doc, use_cache=True).past_key_values)
        view = OrthrusView(target, mask_token_id=63, block_size=4, gradient_checkpointing=True).train()
        # Anchor zero exercises an empty clean prefix. The other case has
        # overlapping blocks and non-monotone anchor order within the chunk.
        for positions in (torch.tensor([0]), torch.tensor([19, 1, 4, 16])):
            ids = doc[0, positions]
            full = view(ids, positions, kv)
            cotangent = torch.randn_like(full)
            full_grads = torch.autograd.grad(full, tuple(view.parameters()), cotangent)
            bound = int(positions.max())
            widths = []
            sdpa = F.scaled_dot_product_attention

            def record_width(q, k, v, **kwargs):
                widths.append(k.shape[-2])
                return sdpa(q, k, v, **kwargs)

            with patch.object(F, "scaled_dot_product_attention", side_effect=record_width):
                trimmed = view(ids, positions, kv, max_context_length=bound)
                trimmed_grads = torch.autograd.grad(trimmed, tuple(view.parameters()), cotangent)
            self.assertEqual(set(widths), {bound + positions.numel() * view.block_size})
            torch.testing.assert_close(trimmed, full, rtol=2e-5, atol=2e-5)
            for actual, expected in zip(trimmed_grads, full_grads, strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            self.assertEqual(kv[0][0].shape[2], doc.shape[1])


if __name__ == "__main__":
    unittest.main()
