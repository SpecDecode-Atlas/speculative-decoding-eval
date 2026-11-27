"""Full-vocabulary distillation with bounded vocabulary-head activation memory."""
import torch
from torch.nn import functional as F


def _teacher_targets(teacher_logits):
    teacher_logits = teacher_logits.detach()
    labels = teacher_logits.argmax(-1)
    p = F.softmax(teacher_logits.float(), dim=-1)
    return labels, p


def _student_terms(logits, labels, p):
    logq = F.log_softmax(logits.float(), dim=-1)
    ce = F.nll_loss(logq, labels, reduction="none")
    tv = (p - logq.exp()).abs().sum(dim=-1) * 0.5
    return ce, tv


def distribution_terms(logits, teacher_logits):
    """Use a teacher argmax label for CE and the full teacher distribution for TV."""
    return _student_terms(logits, *_teacher_targets(teacher_logits))


def hidden_gradient(hidden, teacher_hidden, weights, head, chunk_size, loss_scale, *, teacher_positions=None):
    """Compute the exact gradient entering the proposal network one head chunk at a time.

    Detaching the vocabulary-head input is safe because that head is frozen.
    The returned hidden gradient is passed through the original adapter once.
    Supply CPU teacher_positions to index into one document's teacher_hidden.
    The frozen projection and softmax only run once per distinct document
    offset within a chunk. Student rows retain their individual loss weights.
    """
    shape = hidden.shape
    flat = hidden.reshape(-1, shape[-1])
    if teacher_positions is None:
        teacher = teacher_hidden.reshape_as(flat)
    else:
        teacher = teacher_hidden.reshape(-1, shape[-1])
        if teacher_positions.device.type != "cpu" or teacher_positions.shape != weights.shape:
            raise ValueError("Expected a CPU teacher position for each weighted student row")
        teacher_positions = teacher_positions.reshape(-1)
    weights = weights.reshape(-1)
    result = torch.empty_like(flat)
    stats = torch.zeros(2, device=hidden.device, dtype=torch.float32)
    for start in range(0, flat.shape[0], chunk_size):
        sl = slice(start, start + chunk_size)
        leaf = flat[sl].detach().requires_grad_(True)
        with torch.no_grad():
            order = None
            if teacher_positions is None:
                teacher_input = teacher[sl]
            else:
                positions = teacher_positions[sl]
                unique, inverse = torch.unique(positions, return_inverse=True)
                if unique.numel() < positions.numel():
                    positions = unique
                    order = inverse.to(teacher.device)
                teacher_input = teacher.index_select(0, positions.to(teacher.device))
            labels, p = _teacher_targets(head(teacher_input.to(head.weight.dtype)))
            if order is not None:
                labels = labels.index_select(0, order)
                p = p.index_select(0, order)
        ce, tv = _student_terms(head(leaf.to(head.weight.dtype)), labels, p)
        del p  # The next head chunk can reuse this vocabulary-sized buffer.
        sums = torch.stack(((weights[sl] * ce).sum(), (weights[sl] * tv).sum()))
        objective = (0.1 * sums[0] + 0.9 * sums[1]) * loss_scale
        grad, = torch.autograd.grad(objective, leaf)
        result[sl] = grad
        stats += sums.detach()
    return result.reshape(shape), stats
