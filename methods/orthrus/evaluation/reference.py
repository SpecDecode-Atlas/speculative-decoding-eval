"""Generate against one frozen Qwen3 model with a single mutable target cache."""

from dataclasses import dataclass

import torch


@dataclass
class Decode:
    tokens: list[int]
    rounds: int = 0
    accepted_draft: int = 0
    proposed: int = 0
    draft_passes: int = 0
    verify_passes: int = 0

    @property
    def committed_after_prefill(self):
        return max(0, len(self.tokens) - 1)


def verify_block(proposals, target_predictions, remaining, eos_ids):
    """Return newly emitted IDs and number drawn from the agreeing draft prefix.

    Row j of target_predictions is conditioned on the anchor and proposals[:j].
    Its last row supplies the bonus when every proposal agrees.
    """
    if remaining < 0 or len(target_predictions) != len(proposals) + 1:
        raise ValueError("Expected one target row per proposal plus a bonus row")
    committed = []
    accepted = 0
    for prediction in target_predictions[:remaining]:
        committed.append(prediction)
        match = accepted < len(proposals) and proposals[accepted] == prediction
        if match:
            accepted += 1
        if not match or prediction in eos_ids:
            break
    return committed, accepted


def _prefill(target, prompt_ids, max_new_tokens):
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("Expected a nonempty [1,prompt_length] tensor")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    out = target(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    token = out.logits[0, -1].argmax().item()
    return Decode([token]), out.past_key_values


@torch.inference_mode()
def ar_decode(target, prompt_ids, max_new_tokens, eos_ids):
    result, cache = _prefill(target, prompt_ids, max_new_tokens)
    while len(result.tokens) < max_new_tokens and result.tokens[-1] not in eos_ids:
        anchor = prompt_ids.new_tensor([[result.tokens[-1]]])
        out = target(input_ids=anchor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        result.tokens.append(out.logits[0, -1].argmax().item())
        result.rounds += 1
        result.verify_passes += 1
    return result


@torch.inference_mode()
def orthrus_decode(target, view, prompt_ids, max_new_tokens, eos_ids):
    from orthrus_atlas import cache_layers

    result, cache = _prefill(target, prompt_ids, max_new_tokens)
    while len(result.tokens) < max_new_tokens and result.tokens[-1] not in eos_ids:
        # The last emitted token is known, but its K/V is not in the cache yet.
        prefix = prompt_ids.shape[1] + len(result.tokens) - 1
        remaining = max_new_tokens - len(result.tokens)
        anchor = prompt_ids.new_tensor([result.tokens[-1]])
        position = prompt_ids.new_tensor([prefix])
        room = target.config.max_position_embeddings - prefix
        if room < view.block_size:
            hidden = view(anchor, position, cache_layers(cache), block_size=room)
        else:
            hidden = view(anchor, position, cache_layers(cache))
        proposals = target.lm_head(hidden[:, :remaining]).argmax(dim=-1)
        block = torch.cat((anchor.view(1, 1), proposals), dim=1)
        out = target(input_ids=block, past_key_values=cache, use_cache=True)
        target_tokens = out.logits[0].argmax(dim=-1)
        # One host transfer per round. Retain only the contiguous matching prefix.
        both = torch.cat((proposals[0], target_tokens)).tolist()
        width = proposals.shape[1]
        committed, accepted = verify_block(both[:width], both[width:], remaining, eos_ids)
        cache = out.past_key_values
        cache.crop(prefix + 1 + accepted)
        result.tokens.extend(committed)
        result.rounds += 1
        result.accepted_draft += accepted
        result.proposed += width
        result.draft_passes += 1
        result.verify_passes += 1
    return result
