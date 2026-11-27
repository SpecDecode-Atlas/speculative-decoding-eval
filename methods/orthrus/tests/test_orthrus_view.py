"""Small, model-download-free tests of the Orthrus view on a real Qwen3 decoder."""
import copy
import io
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from orthrus_atlas import OrthrusView, cache_layers


def tiny_target():
    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, attention_dropout=0.0, tie_word_embeddings=True,
        eos_token_id=126, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval().requires_grad_(False)


class ViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(303)
        self.target = tiny_target()
        self.view = OrthrusView(self.target, mask_token_id=127, block_size=4).eval()
        self.doc = torch.tensor([[2, 4, 11, 7, 31, 32, 5, 8, 9, 23, 19]])
        self.kv = self.cache(self.doc)

    @torch.no_grad()
    def cache(self, doc):
        return tuple(cache_layers(self.target.model(input_ids=doc, use_cache=True).past_key_values))

    def draft(self, pos, kv):
        pos = torch.tensor(pos)
        return self.view(self.doc[0, pos], pos, kv)

    def test_native_layer_number_and_tensor_alignment(self):
        """A cache flattened at the wrong layer can pass a tensor shape test."""
        shortened = self.cache(self.doc[:, :5])
        self.assertEqual(len(self.kv), self.target.config.num_hidden_layers)
        for index, ((k, v), (short_k, short_v)) in enumerate(zip(self.kv, shortened)):
            self.assertEqual(tuple(k.shape), (1, 2, self.doc.shape[1], 8))
            self.assertEqual(k.shape, v.shape)
            torch.testing.assert_close(k[:, :, :5], short_k, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(v[:, :, :5], short_v, atol=2e-5, rtol=1e-5)
            if index:
                self.assertFalse(torch.allclose(k, self.kv[index - 1][0]))

    @torch.no_grad()
    def test_initial_view_matches_native_decoder_with_bidirectional_block(self):
        # The copied diffusion projections equal the untouched AR projections at
        # step zero. Feed the real HF decoder an explicit all-visible block
        # attention mask and a separately prefetched causal prefix as an oracle.
        anchor, width = 4, 4
        prefix = self.target.model(input_ids=self.doc[:, :anchor], use_cache=True).past_key_values
        ids = torch.tensor([[self.doc[0, anchor].item(), 127, 127, 127]])
        positions = torch.arange(anchor, anchor + width).unsqueeze(0)
        out = self.target.model(
            input_ids=ids, past_key_values=prefix, use_cache=True, position_ids=positions,
            attention_mask=torch.zeros(1, 1, width, anchor + width),
        ).last_hidden_state
        torch.testing.assert_close(self.draft([anchor], self.kv), out[:, :-1], atol=2e-5, rtol=1e-5)

    @torch.no_grad()
    def test_future_and_anchor_teacher_kv_are_invisible(self):
        """The known anchor is embedded by the diffusion path, not read from target KV."""
        a = 4
        reference = self.draft([a], self.cache(self.doc[:, :a]))
        full_output = self.draft([a], self.kv)
        torch.testing.assert_close(reference, full_output, atol=2e-5, rtol=1e-5)
        poison = [(k.clone(), v.clone()) for k, v in self.kv]
        for k, v in poison:
            k[:, :, a:].fill_(201.0)
            v[:, :, a:].fill_(-100.0)
        torch.testing.assert_close(reference, self.draft([a], poison), atol=2e-5, rtol=1e-5)

    @torch.no_grad()
    def test_every_prefix_layer_contributes_and_blocks_do_not_interact(self):
        together = self.draft([3, 7], self.kv)
        self.assertEqual(tuple(together.shape), (2, 3, 32))
        independent = torch.cat([self.draft([a], self.kv) for a in (3, 7)])
        torch.testing.assert_close(together, independent, atol=2e-5, rtol=1e-5)
        for index, (k, v) in enumerate(self.kv):
            modified = list(self.kv)
            modified[index] = (k, v + 7.0)
            self.assertFalse(torch.allclose(self.draft([3], modified), independent[:1]))

    def test_frozen_target_gradient_and_portable_adapter_reload(self):
        initial_target = {name: p.clone() for name, p in self.target.named_parameters()}
        before = self.draft([3, 7], self.kv).detach()
        optimizer = torch.optim.AdamW([p for p in self.view.parameters() if p.requires_grad], lr=0.001)
        labels = torch.tensor([16, 2, 37, 3, 17, 9])
        logits = self.target.lm_head(self.draft([3, 7], self.kv))
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.view.parameters()))
        self.assertTrue(all(p.grad is None for p in self.target.parameters()))
        optimizer.step()
        for name, p in self.target.named_parameters():
            torch.testing.assert_close(initial_target[name], p, atol=0, rtol=0)
        after = self.draft([3, 7], self.kv).detach()
        self.assertFalse(torch.equal(before, after))
        adapter_bytes = io.BytesIO()
        torch.save(self.view.state_dict(), adapter_bytes)
        adapter_bytes.seek(0)
        fresh_target = copy.deepcopy(self.target)
        restored = OrthrusView(fresh_target, mask_token_id=127, block_size=4).eval()
        restored.load_state_dict(torch.load(adapter_bytes, map_location="cpu", weights_only=True))
        positions = torch.tensor([3, 7])
        torch.testing.assert_close(restored(self.doc[0, positions], positions, self.kv), after)
