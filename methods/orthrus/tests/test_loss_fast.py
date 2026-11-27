"""Repeated teacher offsets preserve per-proposal losses and gradients."""
import unittest

import torch

from training.objective import hidden_gradient


class VocabularyProjectionTests(unittest.TestCase):
    def test_overlapping_blocks_share_only_frozen_projections(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(234)
                head = torch.nn.Linear(32, 257, dtype=dtype).requires_grad_(False)
                document = torch.randn(12, 32, dtype=dtype)
                # The target sees the same offsets in different draft blocks.
                # The last block also mimics positions clamped at a short tail.
                positions = torch.tensor([[3, 4, 5, 6], [4, 5, 6, 7], [7, 8, 8, 8]])
                hidden = torch.randn(3, 4, 32, dtype=dtype)
                weights = torch.tensor([[1., .8, .6, 0.], [.2, 1., 0., .3], [1., .1, 0., 0.]])
                expected, expected_sums = hidden_gradient(
                    hidden, document[positions], weights, head, 8, 0.45)
                calls = []
                handle = head.register_forward_pre_hook(
                    lambda _, args: calls.append((torch.is_grad_enabled(), len(args[0]))))
                try:
                    grad, sums = hidden_gradient(hidden, document, weights, head, 8,
                                                 0.45, teacher_positions=positions)
                finally:
                    handle.remove()
                torch.testing.assert_close(grad.float(), expected.float(), rtol=0.01, atol=1e-6)
                torch.testing.assert_close(sums, expected_sums)
                self.assertEqual([size for train, size in calls if not train], [5, 2])
                self.assertEqual([size for train, size in calls if train], [8, 4])
                self.assertEqual(grad[weights == 0].count_nonzero().item(), 0)

    def test_distinct_offsets_keep_their_original_order(self):
        head = torch.nn.Linear(7, 19).requires_grad_(False)
        positions = torch.tensor([5, 0, 3, 1])
        document = torch.randn(6, 7)
        # Identical target and draft hidden rows exercise the TV equality case.
        hidden = document[positions]
        weights = torch.tensor([.4, 1., 0., .1])
        expected = hidden_gradient(hidden, hidden, weights, head, 3, 0.5)
        actual = hidden_gradient(hidden, document, weights, head, 3, 0.5,
                                 teacher_positions=positions)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference)
