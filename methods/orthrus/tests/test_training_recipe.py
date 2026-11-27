"""Measure the training loss, packed sampling, and resume against independent oracles."""
import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training import checkpoint
from training.data import append_tail, local_plan, sample_blocks, slot_weights
from training.objective import hidden_gradient
from training.train import learning_rate, train_step
from orthrus_atlas import OrthrusView, cache_layers
from test_orthrus_view import tiny_target


class TrainingTests(unittest.TestCase):
    def test_chunked_head_gradient_matches_dense_ce_plus_tv(self):
        torch.manual_seed(273)
        student = torch.randn(2, 3, 8, requires_grad=True)
        teacher = torch.randn_like(student)
        head = torch.nn.Linear(8, 11, bias=False).requires_grad_(False)
        weights = torch.tensor([[1.0, 0.4, 0.0], [0.1, 0.2, 0.9]])
        scale = 1 / weights.sum()
        student_dist = head(student).softmax(-1)
        teacher_dist = head(teacher).softmax(-1)
        ce = -student_dist.gather(-1, teacher_dist.argmax(-1, keepdim=True)).squeeze(-1).log()
        tv = 0.5 * (student_dist - teacher_dist).abs().sum(-1)
        weighted = torch.stack([(ce * weights).sum(), (tv * weights).sum()])
        expected, = torch.autograd.grad((0.1 * weighted[0] + 0.9 * weighted[1]) * scale, student)
        for chunk in (1, 2, 5, 30):
            gradient, sums = hidden_gradient(student, teacher, weights, head, chunk, scale)
            torch.testing.assert_close(gradient, expected)
            torch.testing.assert_close(sums, weighted)

    def test_actual_optimizer_step_matches_independent_two_document_distillation(self):
        torch.manual_seed(208)
        torch.set_num_threads(1)
        target = tiny_target()
        native = OrthrusView(target, mask_token_id=127, block_size=4)
        manual = OrthrusView(target, mask_token_id=127, block_size=4)
        manual.load_state_dict(copy.deepcopy(native.state_dict()))
        rows = [
            {"row": 3, "ids": torch.tensor([8, 77, 3, 19, 44]), "weights": torch.tensor([0., 0., 1., 0.5, 1.])},
            {"row": 7, "ids": torch.tensor([1, 9, 5]), "weights": torch.tensor([0., 1., 1.])},
        ]
        args = SimpleNamespace(max_anchors=99, blocks_per_chunk=2, head_chunk_size=2,
                               block_size=4, seed=42, smoke=False)
        choose_optimizer = lambda model: torch.optim.SGD(model.parameters(), lr=0.02)
        stats = train_step(native, target, choose_optimizer(native), None, rows, args, 1, 0, 1, torch.device("cpu"))
        summed = 0.0
        denominator = 0.0
        for row in rows:
            ids, weights = row["ids"], row["weights"]
            with torch.no_grad():
                teacher = target.model(ids[None, :], use_cache=True)
                teacher_logits = target.lm_head(teacher.last_hidden_state[0])
            for a in range(len(ids) - 1):
                if not weights[a + 1]:
                    continue
                h = manual(ids[a:a + 1], torch.tensor([a]), cache_layers(teacher.past_key_values))
                predictions = target.lm_head(h[0])
                for index in range(min(3, len(ids) - a - 1)):
                    slot_weight = weights[a + index + 1].item() * math.exp(-index / 7.0)
                    p = teacher_logits[a + index].softmax(-1)
                    logq = predictions[index].log_softmax(-1)
                    ce = -logq[p.argmax()]
                    tv = (0.5 * (p - logq.exp()).abs()).sum()
                    summed = summed + slot_weight * (0.1 * ce + 0.9 * tv)
                    denominator += slot_weight
        loss = summed / denominator
        optimizer = choose_optimizer(manual)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(manual.parameters(), 1.0)
        optimizer.step()
        self.assertAlmostEqual(stats["weight"], denominator, places=5)
        self.assertAlmostEqual(stats["loss"], loss.item(), places=5)
        for p, q in zip(native.parameters(), manual.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)

    def test_pack_anchor_budget_and_future_slot_masks(self):
        rows = [
            {"ids": torch.arange(6), "weights": torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.float32)},
            {"ids": torch.arange(3), "weights": torch.tensor([0, 1, 1], dtype=torch.float32)},
        ]
        all_starts = sample_blocks(rows, max_anchors=100, seed=42)
        self.assertEqual([a.tolist() for a in all_starts], [[1, 2, 4], [0, 1]])
        chosen = sample_blocks(rows, max_anchors=3, seed=42)
        self.assertEqual(sum(v.numel() for v in chosen), 3)
        self.assertEqual([v.tolist() for v in chosen], [v.tolist() for v in sample_blocks(rows, 3, 42)])
        for a, w in zip(chosen, all_starts):
            self.assertEqual(len(set(a.tolist())), len(a))
            self.assertTrue(set(a.tolist()).issubset(set(w.tolist())))
        slots = slot_weights(rows[0], torch.tensor([2, 4]), proposals=4, decay=7.0)
        expected = torch.tensor([[1.0, 0.0, math.exp(-2 / 7), 0.0], [1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(slots, expected)

    def test_one_epoch_distributed_tail_has_no_missing_or_duplicated_rows(self):
        rows = 100 + np.arange(8)
        lengths = np.array([6] * len(rows))
        world, budget = 3, 7
        plans = [local_plan(rows, lengths, budget, world, i, 42) for i in range(world)]
        used = sum(len(batch) for plan in plans for batch in plan)
        self.assertEqual(used, 6)
        for rank, plan in enumerate(plans):
            append_tail(plan, rows, used, world, rank, seed=42)
        all_rows = [row for plan in plans for batch in plan for row in batch]
        self.assertEqual(sorted(all_rows), sorted(rows.tolist()))
        self.assertEqual(set(map(len, plans)), {3})
        for plan in plans:
            self.assertTrue(all(6 * len(batch) <= budget for batch in plan))

    def test_learning_rate_warmup_then_cosine(self):
        values = [learning_rate(step, total=150) for step in range(1, 151)]
        np.testing.assert_allclose(values[:50], 0.0006 * np.arange(1, 51) / 50)
        self.assertTrue(all(a >= b >= 0.0 for a, b in zip(values[50:], values[51:])))
        self.assertAlmostEqual(values[100], 0.0003)
        self.assertLess(values[-1], 1e-6)
        with self.assertRaises(ValueError):
            learning_rate(0, 150)

    def test_quarter_checkpoints_and_exact_next_cpu_update(self):
        self.assertEqual(checkpoint.quarter_steps(11), [3, 6, 9, 11])
        self.assertEqual(checkpoint.quarter_steps(1), [1])
        torch.manual_seed(140)
        random.seed(140)
        np.random.seed(140)
        device = torch.device("cpu")
        def fresh():
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.01)
            return model, optimizer

        def update(model, optimizer):
            x = torch.randn(7, 3) + random.random() + np.random.uniform()
            y = torch.randn(7, 2)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return loss.item()

        model, opt = fresh()
        update(model, opt)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint.save(root, 1, model, opt, {"test": True}, 0, 1, device)
            saved = checkpoint.read_resume(root / "checkpoints/latest.json")
            expected = update(model, opt)
            restored, restored_opt = fresh()
            restored.load_state_dict(torch.load(saved / "adapter.pt", weights_only=True))
            checkpoint.restore_optimizer(restored, restored_opt, saved, 0, 1)
            checkpoint.restore_rng(saved, 0, device)
            self.assertEqual(update(restored, restored_opt), expected)
            for original, actual in zip(model.parameters(), restored.parameters(), strict=True):
                torch.testing.assert_close(actual, original, atol=0.0, rtol=0.0)
