# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import random
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

from tqdm import tqdm


def generate_connected_masks(
    B: int,
    V: int,
    H: int,
    W: int,
    ratio: float,
    device=None,
    dtype=torch.uint8,
    ar_range=(0.3, 3.0),
    mode: str = "rectangle"
) -> torch.Tensor:
    """
    Generate connected masks for masking image patches.
    Returns: (B, V, H, W), 1=masked, 0=visible.
    """
    device = device or "cpu"
    mask = torch.zeros(B, V, H, W, device=device, dtype=dtype)
    r = ratio

    if mode == "rectangle":
        total = H * W
        for b in range(B):
            for v in range(V):
                target_area = max(1, int(round(r * total)))
                log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
                ar = math.exp(torch.empty(()).uniform_(log_min, log_max).item())
                h = max(1, int(round(math.sqrt(target_area / ar))))
                w = max(1, int(round(ar * h)))
                h = max(1, min(h, H))
                w = max(1, min(w, W))
                top = 0 if H == h else int(torch.randint(0, H - h + 1, (1,)).item())
                left = 0 if W == w else int(torch.randint(0, W - w + 1, (1,)).item())
                mask[b, v, top:top+h, left:left+w] = 1

    elif mode == "random_walk":
        from collections import deque
        for b in range(B):
            for v in range(V):
                target = max(1, int(round(r * H * W)))
                visited = torch.zeros((H, W), device=device, dtype=torch.bool)
                y = int(torch.randint(0, H, (1,)).item())
                x = int(torch.randint(0, W, (1,)).item())
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                filled = 0
                while q and filled < target:
                    cy, cx = q.popleft()
                    mask[b, v, cy, cx] = 1
                    filled += 1
                    if filled >= target:
                        break
                    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                    idx = torch.randperm(4)
                    for i in idx.tolist():
                        dy, dx = dirs[i]
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))

    elif mode == "ellipse":
        total = H * W
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing="ij"
        )
        for b in range(B):
            for v in range(V):
                target_area = max(1, int(round(r * total)))
                base_r = math.sqrt(target_area / math.pi)
                ar = math.exp(torch.empty(()).uniform_(
                    math.log(ar_range[0]), math.log(ar_range[1])
                ).item())
                ry = max(1, int(round(base_r / math.sqrt(ar))))
                rx = max(1, int(round(base_r * math.sqrt(ar))))
                ry, rx = min(ry, H//2), min(rx, W//2)
                cy = int(torch.randint(ry, max(ry+1, H - ry), (1,)).item()) if H - 2*ry > 0 else H // 2
                cx = int(torch.randint(rx, max(rx+1, W - rx), (1,)).item()) if W - 2*rx > 0 else W // 2
                ellipse = ((yy - cy)**2 / (ry**2 + 1e-6) + (xx - cx)**2 / (rx**2 + 1e-6)) <= 1
                mask[b, v] = ellipse.to(dtype)

    elif mode == "blob":
        smooth_k = 10
        total = H * W
        target = max(1, int(round(r * total)))
        noise = torch.randn(B*V, 1, H, W, device=device)
        kernel_size = min(smooth_k, H, W)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = torch.nn.functional.avg_pool2d(noise, kernel_size, stride=1, padding=kernel_size//2)
        flat = smoothed.view(B*V, -1)
        kth = torch.topk(flat, target, dim=-1).values.min(dim=-1, keepdim=True).values
        mask = (flat >= kth).view(B, V, H, W)

    elif mode == "random":
        total = H * W
        target = max(1, int(round(r * total)))
        noise = torch.randn(B*V, 1, H, W, device=device)
        flat = noise.view(B*V, -1)
        kth = torch.topk(flat, target, dim=-1).values.min(dim=-1, keepdim=True).values
        mask = (flat >= kth).view(B, V, H, W)

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return mask.to(dtype)


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        use_masking=False,
        use_mae_masking=False,
        mask_mode=None,
        mask_ratio=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.use_masking = use_masking
        self.use_mae_masking = use_mae_masking
        self.mask_mode = mask_mode if mask_mode is not None else ['random']
        self.mask_ratio = mask_ratio if mask_ratio is not None else [0.75]


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

            # Masked reconstruction support
            self.use_masking = config.use_masking
            self.use_mae_masking = config.use_mae_masking
            self.mask_mode = config.mask_mode
            self.mask_ratio = config.mask_ratio
            if self.use_masking or self.use_mae_masking:
                # Learnable placeholder embedding for masked positions in nonref clean VAE tokens.
                # Note: nonref images have no VIT (excluded at data side), so no VIT placeholder needed.
                self.vae_mask_placeholder = nn.Parameter(torch.zeros(self.hidden_size))

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)
            if self.use_masking:
                nn.init.normal_(self.vae_mask_placeholder, std=0.02)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_vae_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            if self.use_masking and packed_vae_types is not None:
                # === Masked Reconstruction Mode ===
                #
                # Data layout uses explicit vae_type labels from the dataset:
                #   "ref"          — ref clean VAE (Pass 1 ref images, no loss)
                #   "nonref_clean" — nonref clean VAE (Pass 1 nonref images, no loss)
                #   "nonref_noise" — nonref noise VAE (Pass 2 nonref images, with loss)
                #
                # Strategy:
                # 1. Classify VAE images by packed_vae_types labels
                # 2. Generate spatial masks for each nonref image
                # 3. Noise VAE: fully noised via flow-matching (entire image, not just masked)
                #    — the model sees visible parts through the nonref clean VAE condition
                # 4. Nonref clean VAE: replace masked positions with vae_mask_placeholder
                #
                # Note: nonref images have NO VIT tokens (removed at data side to prevent
                # information leakage). Only ref images have VIT tokens.

                i = random.randint(0, len(self.mask_mode) - 1)
                cur_mode = self.mask_mode[i]
                cur_ratio = self.mask_ratio[i]

                # --- Step 1: Classify VAE images by explicit vae_type labels ---
                # Each entry: (vae_index, h, w, token_offset)
                ref_vae_list = []          # vae_type == "ref"
                nonref_clean_vae_list = [] # vae_type == "nonref_clean"
                noise_vae_list = []        # vae_type == "nonref_noise"
                token_offset = 0
                for vae_idx, (h, w) in enumerate(patchified_vae_latent_shapes):
                    num_tokens = h * w
                    # All tokens of one VAE image share the same vae_type
                    vae_type = packed_vae_types[token_offset]
                    entry = (vae_idx, h, w, token_offset)
                    if vae_type == "ref":
                        ref_vae_list.append(entry)
                    elif vae_type == "nonref_clean":
                        nonref_clean_vae_list.append(entry)
                    elif vae_type == "nonref_noise":
                        noise_vae_list.append(entry)
                    else:
                        raise ValueError(f"Unknown vae_type: {vae_type!r}. "
                                        f"Expected 'ref', 'nonref_clean', or 'nonref_noise'.") 
                    token_offset += num_tokens

                num_nonref = len(noise_vae_list)
                num_ref = len(ref_vae_list)

                assert len(nonref_clean_vae_list) == num_nonref, (
                    f"nonref_clean ({len(nonref_clean_vae_list)}) and nonref_noise ({num_nonref}) "
                    f"count mismatch — each nonref image must have exactly one clean and one noise entry."
                )
                for i in range(num_nonref):
                    _, hc, wc, _ = nonref_clean_vae_list[i]
                    _, hn, wn, _ = noise_vae_list[i]
                    assert (hc, wc) == (hn, wn), (
                        f"nonref image {i}: clean shape ({hc},{wc}) != noise shape ({hn},{wn})"
                    )

                # --- Step 2: Generate masks for each nonref image ---
                # masks_2d[i]: (h, w) binary mask for nonref image i, 1=masked
                masks_2d = []
                for i in range(num_nonref):
                    _, h, w, _ = noise_vae_list[i]
                    mask = generate_connected_masks(
                        B=1, V=1, H=h, W=w,
                        ratio=cur_ratio,
                        device=packed_latent_clean.device,
                        dtype=torch.uint8,
                        mode=cur_mode,
                    )  # (1, 1, h, w)
                    masks_2d.append(mask.squeeze(0).squeeze(0))  # (h, w)

                # --- Step 3: Build noise VAE flag (all tokens of noise images) ---
                # Noise VAE tokens are fully noised — no per-patch masking needed,
                # because the nonref clean VAE already provides visible patches as condition.
                packed_noise_vae_flags = torch.zeros(
                    packed_latent_clean.shape[0], dtype=torch.bool,
                    device=packed_latent_clean.device
                )
                for i in range(num_nonref):
                    _, h, w, offset = noise_vae_list[i]
                    num_tokens = h * w
                    packed_noise_vae_flags[offset:offset + num_tokens] = True

                # --- Step 4: Build mask flags for nonref clean VAE tokens ---
                packed_clean_vae_mask_flags = torch.zeros(
                    packed_latent_clean.shape[0], dtype=torch.bool,
                    device=packed_latent_clean.device
                )
                for i in range(num_nonref):
                    _, h, w, offset = nonref_clean_vae_list[i]
                    num_tokens = h * w
                    mask_flat = masks_2d[i].flatten().bool()
                    packed_clean_vae_mask_flags[offset:offset + num_tokens] = mask_flat

                # --- Step 4b: Build mask flags for noise VAE tokens (masked positions only) ---
                # This marks which tokens *within* the noise VAE correspond to masked patches.
                # Used later to filter loss computation to masked positions only.
                packed_noise_vae_mask_flags = torch.zeros(
                    packed_latent_clean.shape[0], dtype=torch.bool,
                    device=packed_latent_clean.device
                )
                for i in range(num_nonref):
                    _, h, w, offset = noise_vae_list[i]
                    num_tokens = h * w
                    mask_flat = masks_2d[i].flatten().bool()
                    packed_noise_vae_mask_flags[offset:offset + num_tokens] = mask_flat

                # --- Step 5: Process noise VAE tokens (fully noised via flow-matching) ---
                # Each noise image independently samples its own timestep.
                packed_latent_input = packed_latent_clean.clone()
                noise = torch.randn_like(packed_latent_clean)

                per_token_timesteps = torch.zeros(
                    packed_latent_clean.shape[0], device=packed_latent_clean.device
                )

                for i in range(num_nonref):
                    _, h, w, offset = noise_vae_list[i]
                    num_tokens = h * w

                    # Sample an independent timestep for this noise image
                    raw_t = torch.randn(1, device=packed_latent_clean.device)
                    sampled_t = torch.sigmoid(raw_t)
                    sampled_t = self.timestep_shift * sampled_t / (1 + (self.timestep_shift - 1) * sampled_t)
                    sampled_t = sampled_t.item()

                    # x_t = (1 - t) * clean + t * noise
                    packed_latent_input[offset:offset + num_tokens] = (
                        (1 - sampled_t) * packed_latent_clean[offset:offset + num_tokens]
                        + sampled_t * noise[offset:offset + num_tokens]
                    )
                    per_token_timesteps[offset:offset + num_tokens] = sampled_t

                packed_timestep_embeds = self.time_embedder(per_token_timesteps)
                latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                packed_latent_embedded = self.vae2llm(packed_latent_input) + packed_timestep_embeds + latent_token_pos_emb

                # --- Step 6: Replace masked positions in nonref clean VAE with placeholder ---
                packed_latent_embedded[packed_clean_vae_mask_flags] = self.vae_mask_placeholder

                packed_sequence[packed_vae_token_indexes] = packed_latent_embedded

                # NOTE: Nonref images have NO VIT tokens — this is enforced at the data side
                # (need_vit=False for nonref images in MaskedReconIterableDataset).
                # Only ref images have VIT, preventing information leakage of masked regions.
            elif self.use_mae_masking and packed_vae_types is not None:
                # === Masked Reconstruction Mode ===
                #
                # Data layout uses explicit vae_type labels from the dataset:
                #   "ref"          — ref clean VAE (Pass 1 ref images, no loss)
                #   "nonref_clean" — nonref clean VAE (Pass 1 nonref images, no loss)
                #   "nonref_noise" — nonref noise VAE (Pass 2 nonref images, with loss)
                #
                # Strategy:
                # 1. Classify VAE images by packed_vae_types labels
                # 2. Generate spatial masks for each nonref image
                # 3. Noise VAE: fully noised via flow-matching (entire image, not just masked)
                #    — the model sees visible parts through the nonref clean VAE condition
                # 4. Nonref clean VAE: replace masked positions with vae_mask_placeholder
                #
                # Note: nonref images have NO VIT tokens (removed at data side to prevent
                # information leakage). Only ref images have VIT tokens.

                i = random.randint(0, len(self.mask_mode) - 1)
                cur_mode = self.mask_mode[i]
                cur_ratio = self.mask_ratio[i]

                # --- Step 1: Classify VAE images by explicit vae_type labels ---
                # Each entry: (vae_index, h, w, token_offset)
                ref_noise_vae_list = []    # vae_type == "ref_noise"
                nonref_noise_vae_list = []        # vae_type == "nonref_noise"
                token_offset = 0
                for vae_idx, (h, w) in enumerate(patchified_vae_latent_shapes):
                    num_tokens = h * w
                    # All tokens of one VAE image share the same vae_type
                    vae_type = packed_vae_types[token_offset]
                    entry = (vae_idx, h, w, token_offset)
                    if vae_type == "ref_noise":
                        ref_noise_vae_list.append(entry)
                    elif vae_type == "nonref_noise":
                        nonref_noise_vae_list.append(entry)
                    else:
                        raise ValueError(f"Unknown vae_type: {vae_type!r}. "
                                        f"Expected 'ref', 'ref_noise', 'nonref_clean', or 'nonref_noise'.") 
                    token_offset += num_tokens

                num_nonref = len(nonref_noise_vae_list)
                num_ref = len(ref_noise_vae_list)

                # --- Step 2: Generate masks for each nonref image ---
                # masks_2d[i]: (h, w) binary mask for nonref image i, 1=masked
                masks_2d = []
                for i in range(num_nonref):
                    _, h, w, _ = nonref_noise_vae_list[i]
                    mask = generate_connected_masks(
                        B=1, V=1, H=h, W=w,
                        ratio=cur_ratio,
                        device=packed_latent_clean.device,
                        dtype=torch.uint8,
                        mode=cur_mode,
                    )  # (1, 1, h, w)
                    masks_2d.append(mask.squeeze(0).squeeze(0))  # (h, w)

                # --- Step 4: Build mask flags for noise VAE tokens (masked positions only) ---
                # This marks which tokens *within* the noise VAE correspond to masked patches.
                # Used later to filter loss computation to masked positions only.
                packed_noise_vae_mask_flags = torch.zeros(
                    packed_latent_clean.shape[0], dtype=torch.bool,
                    device=packed_latent_clean.device
                )
                for i in range(num_nonref):
                    _, h, w, offset = nonref_noise_vae_list[i]
                    num_tokens = h * w
                    mask_flat = masks_2d[i].flatten().bool()
                    packed_noise_vae_mask_flags[offset:offset + num_tokens] = mask_flat

                # --- Step 5: Flow-matching 加噪 ---
                noise = torch.randn_like(packed_latent_clean)

                packed_timesteps = torch.sigmoid(packed_timesteps)
                packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
                packed_latent_noised = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
                packed_timestep_embeds = self.time_embedder(packed_timesteps)
                latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                packed_latent = self.vae2llm(packed_latent_noised) + packed_timestep_embeds + latent_token_pos_emb

                # --- Step 6: 在 LLM embedding 空间替换 nonref masked 位置为 learnable placeholder ---
                # vae_mask_placeholder 维度是 hidden_size (3584)，必须在 vae2llm 之后替换。
                # 这样 masked 位置在 LLM 输入中看到的是 placeholder（而非加噪 latent），
                # 模型通过 attention 从 ref 和 nonref unmasked 区域获取信息来重建。
                # 注意：packed_latent_noised 保持完整（包含 masked 位置的真实加噪值），
                # 用于后续 loss 计算中反推 x_0_pred。
                packed_latent[packed_noise_vae_mask_flags] = self.vae_mask_placeholder

                packed_sequence[packed_vae_token_indexes] = packed_latent

                # NOTE: Nonref images have NO VIT tokens — this is enforced at the data side
                # (need_vit=False for nonref images in MaskedReconIterableDataset).
                # Only ref images have VIT, preventing information leakage of masked regions.
            else:
                # === Original Flow-Matching Mode ===
                noise = torch.randn_like(packed_latent_clean)
                packed_timesteps = torch.sigmoid(packed_timesteps)
                packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
                packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
                packed_timestep_embeds = self.time_embedder(packed_timesteps)
                latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
                packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            if self.use_masking and packed_vae_types is not None:
                # === Masked Reconstruction Loss (masked positions only) ===
                # Compute loss only on MASKED tokens of noise VAE images.
                # The entire noise VAE is flow-matched, but we only supervise masked patches.
                #
                # mse_loss_indexes selects all noise VAE tokens in packed_sequence space.
                # We then further filter by noise_internal_mask_flags to keep only masked ones.

                # Step 1: Get predictions for ALL noise VAE tokens
                all_noise_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])

                # Step 2: Build internal mask to select masked positions within noise VAE tokens
                # packed_noise_vae_mask_flags marks masked positions in packed_latent space;
                # packed_noise_vae_flags marks all noise VAE positions in packed_latent space.
                # We need the mask relative to noise-only tokens.
                noise_internal_mask_flags = packed_noise_vae_mask_flags[packed_noise_vae_flags]

                # Step 3: Filter to masked positions only
                masked_preds = all_noise_preds[noise_internal_mask_flags]

                # Step 4: Velocity target only at masked positions
                velocity_target_all = noise[packed_noise_vae_flags] - packed_latent_clean[packed_noise_vae_flags]
                velocity_target = velocity_target_all[noise_internal_mask_flags]

                mse = (masked_preds - velocity_target) ** 2
            elif self.use_mae_masking and packed_vae_types is not None:
                # === MAE-style Masked Reconstruction Loss ===
                # 从 velocity prediction 反推 x_0_pred，然后只在 nonref masked 区域计算 reconstruction loss。
                #
                # 数学推导（flow-matching）：
                #   x_t = (1 - t) * x_0 + t * ε        (x_0=clean, ε=noise)
                #   v_t = ε - x_0                       (velocity target)
                #   => x_0 = x_t - t * v_t
                #
                # 因此：x_0_pred = x_t - t * v_t_pred
                #
                # 索引空间说明：
                #   mse_loss_indexes       — packed_sequence 空间（text + vit + vae）
                #   packed_latent 空间     — 仅 vae token
                # 当前数据侧所有 VAE entry 都是 loss=1（need_vae=False），
                # 因此 mse_loss_indexes 选出的 token 数 == packed_latent_clean.shape[0]，
                # 两者一一对应。x_t / timestep 直接从 packed_latent 空间取即可。
                #
                # Loss 策略（方案 B）：
                #   - ref 图片：被加噪参与 forward，但不算 loss
                #   - nonref 图片：只在 masked 区域算 reconstruction loss
                #   packed_noise_vae_mask_flags 已经只标记了 nonref 的 masked 位置

                # Step 1: 获取所有 VAE token 的 velocity 预测（packed_sequence 空间）
                all_v_t_pred = self.llm2vae(last_hidden_state[mse_loss_indexes])

                # Step 2: 获取对应的 x_t 和 timestep（packed_latent 空间，与 all_v_t_pred 一一对应）
                x_t_all = packed_latent_noised   # 全部 VAE token 的加噪结果
                t_all = packed_timesteps          # 全部 VAE token 的 timestep

                # Step 3: 反推 x_0_pred = x_t - t * v_t_pred
                x_0_pred = x_t_all - t_all[:, None] * all_v_t_pred

                # Step 4: 只在 nonref masked 区域计算 reconstruction loss
                # packed_noise_vae_mask_flags 在 packed_latent 空间中标记 nonref 的 masked 位置
                x_0_pred_masked = x_0_pred[packed_noise_vae_mask_flags]
                x_0_clean_masked = packed_latent_clean[packed_noise_vae_mask_flags]

                mse = (x_0_pred_masked - x_0_clean_masked) ** 2
            else:
                # === Original Flow-Matching Loss ===
                packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
                target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
                has_mse = packed_timesteps > 0
                mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output