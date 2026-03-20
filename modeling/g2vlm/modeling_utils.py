# Copyright (c) 2022 Facebook, Inc. and its affiliates.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: CC BY-NC 4.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under CC BY-NC 4.0, with the full license text
# available at https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

import math

import numpy as np
import torch
from torch import nn
from transformers.activations import ACT2FN

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side ** 2, hidden_size), 
            requires_grad=False
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


class PositionEmbedding_Extra(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side ** 2 + 1, hidden_size), 
            requires_grad=False
        ) # comment +1 for extra tokens 
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side, extra_tokens=1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


# --------------------------------------------------------
# Q-Former for Recon-for-Und
# Following the reference implementation with:
#   - RMSNorm (shared norm for query and target)
#   - 3D Multimodal RoPE on cross-attention Q/K
#   - Gated MLP (SwiGLU style, matching Qwen2 LLM)
#   - GQA (Grouped Query Attention) support
# Two-layer cross-attention architecture:
#   Layer 1: query_tokens attend to question_text_tokens
#   Layer 2: updated_query_tokens attend to dino_hidden_states
# --------------------------------------------------------

def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_multimodal_rotary_pos_emb_single(q, cos, sin, mrope_section, unsqueeze_dim=0):
    """
    Apply 3D multimodal RoPE to a single tensor (query or key).
    Unlike the LLM version which takes (q, k) pair, this operates on one tensor.
    
    Args:
        q: (num_heads, seq_len, head_dim) 
        cos: (3, seq_len, head_dim)
        sin: (3, seq_len, head_dim)
        mrope_section: list of ints for temporal/height/width channel splits
    """
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    return q_embed


def _repeat_kv(hidden_states, n_rep):
    """
    Expand K/V from (num_kv_heads, seq_len, head_dim) to (num_heads, seq_len, head_dim).
    """
    num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_kv_heads * n_rep, slen, head_dim)


def obtain_rotary_pos_id_vision(grid_thw, second_per_grid_ts, spatial_merge_size):
    """Generate 3D position IDs for vision tokens (t, h, w)."""
    if not isinstance(second_per_grid_ts, torch.Tensor):
        second_per_grid_ts = torch.Tensor(second_per_grid_ts).to(device=grid_thw.device)
    position_ids = []
    for i in range(grid_thw.shape[0]):
        t, h, w = grid_thw[i]
        llm_grid_t = t.item()
        llm_grid_h = h.item() // spatial_merge_size
        llm_grid_w = w.item() // spatial_merge_size

        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
        expanded_range = expanded_range.to(second_per_grid_ts.device, second_per_grid_ts.dtype)
        second_per_grid_t = second_per_grid_ts[i]
        time_tensor = expanded_range * second_per_grid_t * 2
        t_index = time_tensor.long().flatten()

        h_index = (
            torch.arange(llm_grid_h)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        ).to(t_index.device, t_index.dtype)
        w_index = (
            torch.arange(llm_grid_w)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        ).to(t_index.device, t_index.dtype)
        position_ids.append(torch.stack([t_index, h_index, w_index]))
    position_ids = torch.cat(position_ids, dim=1)
    return position_ids


def obtain_rotary_pos_id_text(text_lens):
    """Generate 1D position IDs for text tokens (same across t/h/w dims)."""
    position_ids = []
    for item in text_lens:
        position_ids.append(
            torch.arange(item).view(1, -1).expand(3, -1)
        )
    position_ids = torch.cat(position_ids, dim=1)
    return position_ids


def obtain_rotary_pos_id_query(num_samples, num_query_tokens):
    """Generate 1D position IDs for query tokens (same across t/h/w dims)."""
    position_ids = []
    for _ in range(num_samples):
        position_ids.append(
            torch.arange(num_query_tokens).view(1, -1).expand(3, -1)
        )
    position_ids = torch.cat(position_ids, dim=1)
    return position_ids


class QFormerRMSNorm(nn.Module):
    """RMSNorm matching Qwen2 style."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QFormerGatedMLP(nn.Module):
    """Gated MLP (SwiGLU style) matching Qwen2 LLM architecture."""
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu", bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class QFormerCrossAttention(nn.Module):
    """
    Cross-attention with 3D multimodal RoPE and GQA support.
    Follows the reference Qwen2_5_VLCrossAttention implementation.
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, rope_scaling, attention_dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        assert (self.head_dim * self.num_heads) == self.hidden_size, \
            f"hidden_size must be divisible by num_heads (got {self.hidden_size} and {self.num_heads})"

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, query, target, attention_mask, query_position_embeddings, key_position_embeddings):
        """
        Args:
            query:  (q_len, D)
            target: (k_len, D)
            attention_mask: (q_len, k_len) or None
            query_position_embeddings: (cos, sin) for query RoPE
            key_position_embeddings:   (cos, sin) for key RoPE
        Returns:
            (q_len, D)
        """
        q_len = query.size(0)
        k_len = target.size(0)

        query_states = self.q_proj(query)
        key_states = self.k_proj(target)
        value_states = self.v_proj(target)

        # Reshape: (seq_len, D) -> (num_heads, seq_len, head_dim)
        query_states = query_states.view(q_len, -1, self.head_dim).transpose(0, 1)
        key_states = key_states.view(k_len, -1, self.head_dim).transpose(0, 1)
        value_states = value_states.view(k_len, -1, self.head_dim).transpose(0, 1)

        # Apply 3D multimodal RoPE
        cos_q, sin_q = query_position_embeddings
        query_states = _apply_multimodal_rotary_pos_emb_single(
            query_states, cos_q, sin_q, self.rope_scaling["mrope_section"]
        )
        cos_k, sin_k = key_position_embeddings
        key_states = _apply_multimodal_rotary_pos_emb_single(
            key_states, cos_k, sin_k, self.rope_scaling["mrope_section"]
        )

        # GQA: repeat K/V heads to match Q heads
        key_states = _repeat_kv(key_states, self.num_kv_groups)
        value_states = _repeat_kv(value_states, self.num_kv_groups)

        # Compute attention: (num_heads, q_len, k_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
            attn_weights = attn_weights + causal_mask

        # Fix float16 inf issue
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back: (num_heads, q_len, head_dim) -> (q_len, D)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.reshape(q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class QFormerLayer(nn.Module):
    """
    A single Q-Former layer: cross-attention + gated MLP, both with RMSNorm pre-norm.
    Uses shared norm1 for query and target (following reference implementation).
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, 
                 rope_scaling, hidden_act="silu", attention_dropout=0.0):
        super().__init__()
        self.norm1 = QFormerRMSNorm(hidden_size, eps=1e-6)
        self.norm2 = QFormerRMSNorm(hidden_size, eps=1e-6)
        self.attn = QFormerCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_scaling=rope_scaling,
            attention_dropout=attention_dropout,
        )
        self.mlp = QFormerGatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=True,
        )

    def forward(self, query, target, attention_mask, query_position_embeddings, key_position_embeddings):
        """
        Args:
            query:  (q_len, D)
            target: (k_len, D)
            attention_mask: (q_len, k_len)
            query_position_embeddings: (cos, sin)
            key_position_embeddings:   (cos, sin)
        Returns:
            updated query: (q_len, D)
        """
        # Cross-attention with shared norm1
        query = query + self.attn(
            self.norm1(query),
            self.norm1(target),
            attention_mask,
            query_position_embeddings,
            key_position_embeddings,
        )
        # Gated MLP with norm2
        query = query + self.mlp(self.norm2(query))
        return query


class QFormerRotaryEmbedding(nn.Module):
    """
    Rotary position embedding for Q-Former, following Qwen2_5_VL style.
    Handles 3D position_ids (t, h, w) without batch dimension.
    """
    def __init__(self, config):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        self.config = config
        
        # Compute inv_freq
        dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Attention scaling (default 1.0 for standard rope)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: reference tensor for dtype
            position_ids: (3, num_positions) - 3D position indices
        Returns:
            cos, sin: (3, num_positions, head_dim)
        """
        # Expand inv_freq: (3, dim//2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(3, -1, 1)
        # position_ids: (3, num_positions) -> (3, 1, num_positions)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class QFormer(nn.Module):
    """
    Two-layer Q-Former for recon-for-und, following the reference implementation.
    
    Architecture:
        Layer 1: learnable query_tokens cross-attend to question_text_hidden (with RoPE)
        Layer 2: updated_query_tokens cross-attend to dino_hidden_states (with RoPE)
        Final RMSNorm on output
    
    Features:
        - 3D Multimodal RoPE (temporal/height/width) on cross-attention Q/K
        - RMSNorm (shared for query and target in each layer)
        - Gated MLP (SwiGLU, matching Qwen2 LLM)
        - GQA (Grouped Query Attention) support
        - Per-sample attention masking for packed sequences
    """
    def __init__(self, llm_config, num_query_tokens=64):
        """
        Args:
            llm_config: Qwen2VLConfig with hidden_size, num_attention_heads, 
                        num_key_value_heads, intermediate_size, hidden_act,
                        rope_scaling, rope_theta, etc.
            num_query_tokens: number of learnable query tokens per sample
        """
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_size = llm_config.hidden_size

        # Learnable query tokens: (num_query_tokens, hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, llm_config.hidden_size))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Two Q-Former layers
        self.layer1 = QFormerLayer(
            hidden_size=llm_config.hidden_size,
            num_heads=llm_config.num_attention_heads,
            num_kv_heads=llm_config.num_key_value_heads,
            intermediate_size=llm_config.intermediate_size,
            rope_scaling=llm_config.rope_scaling,
            hidden_act=llm_config.hidden_act,
        )
        self.layer2 = QFormerLayer(
            hidden_size=llm_config.hidden_size,
            num_heads=llm_config.num_attention_heads,
            num_kv_heads=llm_config.num_key_value_heads,
            intermediate_size=llm_config.intermediate_size,
            rope_scaling=llm_config.rope_scaling,
            hidden_act=llm_config.hidden_act,
        )
        self.output_norm = QFormerRMSNorm(llm_config.hidden_size)

        # RoPE embedding
        self.rotary_emb = QFormerRotaryEmbedding(config=llm_config)

    def forward(self, question_text_hidden, dino_hidden, text_lengths, 
                dino_grid_thw, second_per_grid_ts, num_samples=None,
                dino_images_per_sample=None):
        """
        Args:
            question_text_hidden: (total_text_tokens, D) - packed question text hidden states
            dino_hidden:          (total_dino_tokens, D) - packed dino hidden states
            text_lengths:         list of int - total question text token count per sample
            dino_grid_thw:        (num_images, 3) - grid dimensions for dino images
            second_per_grid_ts:   temporal scaling factors per image
            num_samples:          int - actual number of samples in the batch
                                  (if None, inferred from len(text_lengths))
            dino_images_per_sample: list of int - number of dino images per sample
                                  (if None, assumes equal distribution)
            
        Returns:
            final_query_tokens: (num_samples * num_query_tokens, D)
        """
        if num_samples is None:
            num_samples = len(text_lengths)
        device = question_text_hidden.device
        nq = self.num_query_tokens

        # Expand learnable queries: (num_samples * nq, D)
        query = self.query_tokens.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, self.hidden_size)

        # ----- Build position IDs -----
        query_position_id = obtain_rotary_pos_id_query(num_samples, nq).to(device, torch.long)
        text_position_id = obtain_rotary_pos_id_text(text_lengths).to(device, torch.long)
        vision_position_id = obtain_rotary_pos_id_vision(
            dino_grid_thw, second_per_grid_ts, spatial_merge_size=1
        ).to(device, torch.long)

        # ----- Compute RoPE embeddings -----
        query_pos_embed = self.rotary_emb(query, query_position_id)
        text_pos_embed = self.rotary_emb(question_text_hidden, text_position_id)
        vision_pos_embed = self.rotary_emb(dino_hidden, vision_position_id)

        # ----- Build attention mask for layer 1 (query x text) -----
        total_text = question_text_hidden.shape[0]
        attention_mask_1 = torch.full((num_samples * nq, total_text), float('-inf'),
                                      device=device, dtype=question_text_hidden.dtype)
        t_begin = 0
        for i, t_length in enumerate(text_lengths):
            attention_mask_1[i * nq : (i + 1) * nq, t_begin : t_begin + t_length] = 0
            t_begin += t_length

        # Layer 1: query attends to question text
        updated_query = self.layer1(
            query, question_text_hidden, attention_mask_1,
            query_pos_embed, text_pos_embed,
        )

        # ----- Build attention mask for layer 2 (query x dino vision) -----
        # Each sample's query tokens should attend to ALL dino images of that sample
        total_vision = dino_hidden.shape[0]
        attention_mask_2 = torch.full((num_samples * nq, total_vision), float('-inf'),
                                      device=device, dtype=dino_hidden.dtype)
        if dino_images_per_sample is None:
            # Fallback: assume equal distribution of images across samples
            num_images = dino_grid_thw.shape[0]
            imgs_per_sample = num_images // num_samples
            dino_images_per_sample = [imgs_per_sample] * num_samples

        v_begin = 0
        img_idx = 0
        for i in range(num_samples):
            n_imgs = dino_images_per_sample[i]
            sample_v_len = 0
            for _ in range(n_imgs):
                sample_v_len += dino_grid_thw[img_idx].prod().item()
                img_idx += 1
            attention_mask_2[i * nq : (i + 1) * nq, v_begin : v_begin + sample_v_len] = 0
            v_begin += sample_v_len

        # Layer 2: updated query attends to dino hidden states
        final_query = self.layer2(
            updated_query, dino_hidden, attention_mask_2,
            query_pos_embed, vision_pos_embed,
        )

        # Final normalization
        final_query = self.output_norm(final_query)

        return final_query  # (num_samples * nq, D)