"""Exact greedy reference checks, including rejection, EOS and truncation."""
import itertools
import unittest

import torch

from evaluation.reference import ar_decode, orthrus_decode, verify_block
from orthrus_atlas import OrthrusView
from test_orthrus_view import tiny_target


class VerificationTests(unittest.TestCase):
    def test_exhaustive_one_round(self):
        # Vocab size 2 and K=4 give 128 possible proposal / verification pairs.
        # Build the oracle from the first disagreement, then truncate at EOS/cap.
        for draft in itertools.product(range(2), repeat=3):
            for target in itertools.product(range(2), repeat=4):
                first_miss = next((j for j in range(len(draft)) if draft[j] != target[j]), len(draft))
                legal = list(target[: first_miss + 1])
                for eos in (set(), {1}):
                    for remaining in (1, 2, 3, 4):
                        expected = legal[:remaining]
                        if eos and 1 in expected:
                            expected = expected[: expected.index(1) + 1]
                        with self.subTest(draft=draft, target=target, eos=eos, remaining=remaining):
                            ids, accepted = verify_block(list(draft), list(target), remaining, eos)
                            self.assertEqual(ids, expected)
                            self.assertEqual(accepted, min(len(expected), first_miss))

    def test_tiny_qwen_generation_agrees_with_independent_full_recompute(self):
        torch.set_num_threads(1)
        torch.manual_seed(901)
        target = tiny_target()
        view = OrthrusView(target, mask_token_id=127, block_size=4).eval()
        prompt = torch.tensor([[3, 19, 21, 8, 47]])
        with torch.inference_mode():
            for cap in (1, 2, 4, 9):
                sequence = prompt
                expected = []
                for _ in range(cap):
                    token = target(input_ids=sequence, use_cache=False).logits[:, -1].argmax(-1)
                    expected.append(token.item())
                    sequence = torch.cat([sequence, token[:, None]], dim=1)
                self.assertEqual(ar_decode(target, prompt, cap, set()).tokens, expected)
                generated = orthrus_decode(target, view, prompt, cap, set())
                self.assertEqual(generated.tokens, expected)
                self.assertEqual(generated.committed_after_prefill, max(0, cap - 1))

    def test_cache_rollback_with_partial_zero_and_full_acceptance(self):
        torch.manual_seed(285)
        torch.set_num_threads(1)
        target = tiny_target()
        prompt = torch.tensor([[22, 118, 17, 11]])
        all_hidden = []
        gold = []
        seq = prompt
        with torch.inference_mode():
            for _ in range(15):
                h = target.model(input_ids=seq, use_cache=False).last_hidden_state[:, -1]
                all_hidden.append(h)
                t = target.lm_head(h).argmax(-1)
                gold.append(t.item())
                seq = torch.cat([seq, t.unsqueeze(1)], dim=1)

        class ControlledDraft(torch.nn.Module):
            block_size = 4

            def __init__(self):
                super().__init__()
                self.accept_plan = iter([2, 0, 3, 0])

            def forward(self, anchor_ids, positions, kv):
                start = positions.item() - prompt.shape[1] + 1
                self.assert_anchor(anchor_ids, start)
                states = torch.stack(all_hidden[start:start + 3], dim=1)
                first_bad = next(self.accept_plan)
                if first_bad < 3:
                    states[:, first_bad] = -states[:, first_bad]
                return states

            def assert_anchor(self, ids, start):
                if ids.item() != gold[start - 1]:
                    raise AssertionError("Verifier's rollback lost the target's AR prefix")

        decoded = orthrus_decode(target, ControlledDraft(), prompt, 10, set())
        self.assertEqual(decoded.tokens, gold[:10])
        self.assertEqual(decoded.rounds, 4)
        self.assertEqual(decoded.accepted_draft, 5)
