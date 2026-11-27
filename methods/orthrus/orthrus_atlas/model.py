"""Orthrus on Qwen3 using PyTorch scaled dot product attention.

The author implementation (chiennv2000/orthrus, src/model.py) adds separate
Q/K/V/O projections and Q/K head normalization at every transformer layer.
We implement that design with the native frozen Hugging Face Qwen3 backbone.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


LayerKV = tuple[Tensor, Tensor]


def cache_layers(cache) -> tuple[LayerKV, ...]:
    """Return references to the post-RoPE K and unexpanded V of each AR layer.

    DynamicCache layouts differ across Transformers versions. This helper
    preserves device, dtype and storage, including the verifier's crop point.
    The cache must contain one unpadded document and use full attention.
    """
    if hasattr(cache, "layers"):
        pairs = ((layer.keys, layer.values) for layer in cache.layers)
    elif hasattr(cache, "key_cache"):
        pairs = zip(cache.key_cache, cache.value_cache, strict=True)
    elif isinstance(cache, (tuple, list)):
        pairs = cache
    else:
        raise TypeError("Expected a populated Hugging Face DynamicCache or layer KV pairs")
    result = tuple((key, value) for key, value in pairs)
    if not result:
        raise ValueError("The target cache contains no layers")
    for key, value in result:
        if not isinstance(key, Tensor) or not isinstance(value, Tensor):
            raise ValueError("Every target layer must supply K and V")
        if key.ndim != 4 or key.shape[0] != 1 or value.shape != key.shape:
            raise ValueError("Expected a single-document target cache with layout [1, KV heads, time, head dim]")
    return result


def _block_mask(anchor_positions: Tensor, block_size: int, prefix_length: int) -> Tensor:
    """SDPA True = allowed. Each query sees its clean prefix and its own noisy block."""
    device = anchor_positions.device
    blocks = anchor_positions.numel()
    history = torch.arange(prefix_length, device=device)[None, :] < anchor_positions[:, None]
    block_ids = torch.arange(blocks, device=device)
    own_block = block_ids[:, None] == block_ids.repeat_interleave(block_size)[None, :]
    # Construct once per forward, and share across heads and layers.
    allowed = torch.cat((history, own_block), dim=-1).repeat_interleave(block_size, dim=0)
    return allowed[None, None, :, :]


class DiffusionLayer(nn.Module):
    """Trainable diffusion attention with the corresponding frozen decoder's MLP."""

    def __init__(self, ar_attention: nn.Module):
        super().__init__()
        self.head_dim = ar_attention.head_dim
        self.groups = ar_attention.num_key_value_groups
        self.scale = ar_attention.scaling
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            copied = deepcopy(getattr(ar_attention, name)).float().requires_grad_(True)
            setattr(self, name, copied)

    def forward(self, hidden: Tensor, layer: nn.Module, kv: LayerKV,
                rotations: tuple[Tensor, Tensor], mask: Tensor) -> Tensor:
        normalized = layer.input_layernorm(hidden)
        shape = (1, hidden.shape[1], -1, self.head_dim)
        q = self.q_proj(normalized).reshape(shape)
        k = self.k_proj(normalized).reshape(shape)
        v = self.v_proj(normalized).reshape(shape).transpose(1, 2)
        # The norm weights are FP32 parameters during training. Cast normalized
        # Q/K back to the activation dtype for mixed-precision SDPA with AR K/V.
        q = self.q_norm(q).to(v.dtype).transpose(1, 2)
        k = self.k_norm(k).to(v.dtype).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *rotations)
        # Batch dimension stays 1: all proposed blocks in this document use
        # the same historical cache. Only the compact GQA head axis is repeated.
        key = repeat_kv(torch.cat((kv[0].detach().to(k.dtype), k), dim=2), self.groups)
        value = repeat_kv(torch.cat((kv[1].detach().to(v.dtype), v), dim=2), self.groups)
        attended = F.scaled_dot_product_attention(
            q, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(1, hidden.shape[1], -1)
        hidden = hidden + self.o_proj(attended)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class OrthrusView(nn.Module):
    """Only diffusion parameters are registered, sharded and checkpointed.

    Move the target to its final device and dtype before constructing the
    view. It will be kept frozen in evaluation mode. Both modules share its
    embedding, layer normalizations, MLPs and final normalization. Apply its
    frozen LM head to the returned hidden states to obtain vocabulary logits.
    """

    def __init__(self, target: nn.Module, mask_token_id: int = 151669,
                 block_size: int = 16, gradient_checkpointing: bool = False):
        super().__init__()
        if getattr(target.config, "model_type", None) != "qwen3":
            raise ValueError("This Orthrus implementation requires a Qwen3 causal LM")
        if block_size < 2:
            raise ValueError("The block must include a known token and at least one proposal")
        decoder = target.model
        if not 0 <= mask_token_id < decoder.embed_tokens.num_embeddings:
            raise ValueError("mask_token_id is outside the frozen target embedding table")
        if any(getattr(layer.self_attn, "sliding_window", None) is not None for layer in decoder.layers):
            raise ValueError("This recipe requires Qwen3 with full attention in every layer")
        target.requires_grad_(False).eval()
        # Keep the decoder alive without registering it as a trainable child.
        # This ensures state_dict and FSDP contain only the additional weights.
        object.__setattr__(self, "_decoder", decoder)
        self.layers = nn.ModuleList(DiffusionLayer(layer.self_attn) for layer in decoder.layers)
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, anchor_ids: Tensor, anchor_positions: Tensor, kv: tuple[LayerKV, ...],
                *, block_size: int | None = None, max_context_length: int | None = None) -> Tensor:
        """Propose from one document, with one or more separately masked blocks.

        An anchor is the known token x[a] at absolute position a. The cache may
        end immediately before it or include the full training document. For
        block size K, output row j (0 <= j < K-1) predicts token x[a+j+1].
        Training can pass the largest anchor in the chunk as a Python integer
        from its CPU sampler. No block reads an AR position at or past this
        boundary; trimming before GQA expansion also saves attention work.
        """
        if anchor_ids.ndim != 1 or anchor_ids.shape != anchor_positions.shape or anchor_ids.numel() == 0:
            raise ValueError("Expected equal nonempty vectors of anchor IDs and absolute positions")
        if len(kv) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} target KV layers, received {len(kv)}")
        size = self.block_size if block_size is None else block_size
        if not 1 <= size <= self.block_size:
            raise ValueError(f"block_size must be between 1 and {self.block_size}")
        decoder = self._decoder
        device = anchor_ids.device
        blocks = anchor_ids.numel()
        offsets = torch.arange(size, device=device)
        positions = (anchor_positions[:, None] + offsets).reshape(1, -1)
        ids = torch.full((blocks, size), self.mask_token_id, dtype=torch.long, device=device)
        ids[:, 0] = anchor_ids
        hidden = decoder.embed_tokens(ids.reshape(1, -1))
        rotations = decoder.rotary_emb(hidden, positions)
        prefix_length = kv[0][0].shape[2]
        if max_context_length is not None:
            if not 0 <= max_context_length <= prefix_length:
                raise ValueError("max_context_length must lie inside the cached document")
            if max_context_length < prefix_length:
                # These are views. No accelerator scalar read or cache copy is needed.
                kv = tuple((k[:, :, :max_context_length], v[:, :, :max_context_length]) for k, v in kv)
                prefix_length = max_context_length
        mask = _block_mask(anchor_positions, size, prefix_length)
        dtype = hidden.dtype
        precision = (torch.autocast(device_type=device.type, dtype=dtype)
                     if dtype in (torch.float16, torch.bfloat16) else nullcontext())
        with precision:
            for diffusion, frozen, context in zip(self.layers, decoder.layers, kv, strict=True):
                if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                    hidden = checkpoint(diffusion, hidden, frozen, context, rotations, mask,
                                        use_reentrant=False, preserve_rng_state=False)
                else:
                    hidden = diffusion(hidden, frozen, context, rotations, mask)
            hidden = decoder.norm(hidden).reshape(blocks, size, -1)
        # The official Qwen3 Orthrus release uses logits[:, :-1] from each block.
        return hidden[:, :-1, :]
