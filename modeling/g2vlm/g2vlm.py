import copy
import math
from typing import List, Tuple, Optional
from typing import Any, Dict, List, Mapping, Optional, Sequence 
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import torchvision

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    get_rope_index_image_3D,
    get_rope_index_image_3D_dino,
    patchify, 
)
from .qwen2vl import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding, PositionEmbedding_Extra
from modeling.g2vlm.ssl_decoder import SSLDecoder
from modeling.pi3.models.layers.transformer_head import Pi3TransformerDecoder, Pi3LinearPts3d, Pi3ContextTransformerDecoder
from modeling.pi3.models.layers.camera_head import Pi3CameraHead
from modeling.pi3.models.layers.pos_embed import RoPE2D, PositionGetter
from modeling.pi3.utils.geometry import homogenize_points
from copy import deepcopy
from easydict import EasyDict

from modeling.pi3.models.pi3_loss import Pi3Loss
from modeling.g2vlm.ssl_decoder import ConfLoss
from data.transforms_vggt import load_and_preprocess_images, load_and_resize14
from tqdm import tqdm
import random
import torch.distributed as dist


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def generate_connected_masks(
    B: int,          # 批次大小
    V: int,          # 视图数量（注意：只是目标视图数，不含参考视图）
    H: int,          # patch 网格高度（如 Hp=32）
    W: int,          # patch 网格宽度（如 Wp=32）
    ratio: float,    # 掩码比例（如 0.9 = 90% 被 mask）
    device=None,
    dtype=torch.uint8,
    ar_range=(0.3, 3.0),  # 长宽比范围（仅部分 mode 使用）
    mode: str = "rectangle"  # 掩码形状模式
) -> torch.Tensor:   # 返回 (B, V, H, W)，1=被mask，0=保留
    """
    Generate connected masks for masking image patches.
    
    Args:
        B: batch size
        V: number of views to mask (excludes reference views)
        H: height in patches
        W: width in patches
        ratio: ratio of patches to mask
        device: device
        dtype: dtype
        mode: masking mode - 'random', 'block', 'column', 'row', 'checkerboard'
    
    Returns:
        mask: (B, V, H*W) binary mask where 1 means masked
    """
    device = device or "cpu"
    mask = torch.zeros(B, V, H, W, device=device, dtype=dtype) # 每个 batch、每个视图、每个 patch 位置的 0/1 掩码
    r = ratio
    # 这几个mask mode的空间连续性逐渐下降
    if mode == "rectangle":
        # ┌────────────────────────────┐
        # │                            │
        # │    ┌──────────────┐        │
        # │    │██████████████│        │
        # │    │██ 被mask区域 █│        │
        # │    │██████████████│        │
        # │    │██████████████│        │
        # │    └──────────────┘        │
        # │                            │
        # └────────────────────────────┘
        # 特点: 连续的矩形区域，随机长宽比，随机位置
        total = H * W
        for b in range(B):
            for v in range(V):

                target_area = max(1, int(round(r * total)))

                log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
                ar = math.exp(torch.empty(()).uniform_(log_min, log_max).item())  # ar = w/h 随机长宽比 0.3~3.0

                h = max(1, int(round(math.sqrt(target_area / ar)))) # 矩形高
                w = max(1, int(round(ar * h))) # 矩形宽

                h = min(h, H)
                w = min(w, W)

                h = max(1, min(h, H))
                w = max(1, min(w, W))

                top = 0 if H == h else int(torch.randint(0, H - h + 1, (1,)).item())
                left = 0 if W == w else int(torch.randint(0, W - w + 1, (1,)).item())
                # 随机放置在 (H, W) 网格中
                mask[b, v, top:top+h, left:left+w] = 1

    elif mode == "random_walk":
        # ┌────────────────────────────┐
        # │                            │
        # │         ██                 │
        # │        ████                │
        # │       ██████               │
        # │        ████████            │
        # │         ██████             │
        # │          ████              │
        # │                            │
        # └────────────────────────────┘
        # 特点: 不规则的连通区域，像水渍扩散
        for b in range(B):
            for v in range(V):

                target = max(1, int(round(r * H * W)))
                visited = torch.zeros((H, W), device=device, dtype=torch.bool)

                y = int(torch.randint(0, H, (1,)).item())
                x = int(torch.randint(0, W, (1,)).item())

                from collections import deque
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
                if filled < target:
                    ys, xs = torch.nonzero(mask[b, v] > 0, as_tuple=True)
                    border = []
                    for yy, xx in zip(ys.tolist(), xs.tolist()):
                        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                            ny, nx = yy + dy, xx + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[b, v, ny, nx] == 0:
                                border.append((ny, nx))
                    if border:
                        border = list(set(border))
                        need = target - filled
                        take = min(need, len(border))
                        idxs = torch.randperm(len(border))[:take].tolist()
                        for k in idxs:
                            ny, nx = border[k]
                            mask[b, v, ny, nx] = 1
    elif mode == "ellipse":
        # ┌────────────────────────────┐
        # │                            │
        # │        ████████            │
        # │      ████████████          │
        # │    ████████████████        │
        # │      ████████████          │
        # │        ████████            │
        # │                            │
        # └────────────────────────────┘
        # 特点: 光滑的椭圆区域，随机长宽比，随机中心
        total = H * W
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing="ij"
        )
        for b in range(B):
            for v in range(V):
                target_area = max(1, int(round(r * total)))
                base_r = math.sqrt(target_area / math.pi)
                ar = math.exp(torch.empty(()).uniform_(math.log(ar_range[0]), math.log(ar_range[1])).item())
                ry = max(1, int(round(base_r / math.sqrt(ar))))
                rx = max(1, int(round(base_r * math.sqrt(ar))))
                ry, rx = min(ry, H//2), min(rx, W//2)

                if H - 2*ry > 0:
                    cy = int(torch.randint(ry, H - ry, (1,)).item())
                else:
                    cy = H // 2
                if W - 2*rx > 0:
                    cx = int(torch.randint(rx, W - rx, (1,)).item())
                else:
                    cx = W // 2

                ellipse = ((yy - cy)**2 / (ry**2 + 1e-6) + (xx - cx)**2 / (rx**2 + 1e-6)) <= 1
                mask[b, v] = ellipse.to(dtype)
    elif mode == "blob":
        # ┌────────────────────────────┐
        # │  ██                        │
        # │ ████    ████               │
        # │  ████  ██████              │
        # │   ██    ████  ██           │
        # │              ████          │
        # │               ██    ██     │
        # │                    ████    │
        # │                     ██     │
        # └────────────────────────────┘
        # 特点: 多个不规则的连通团块，比 random 更聚集
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
        # ┌────────────────────────────┐
        # │█ █  ██ █ █  █ ██ █  █ █ ██│
        # │ ██ █  ██ █ ██  █ ██ █ ██ █│
        # │█ █ ██ █  ██ █ █  ██  █ █ █│
        # │ ██  █ ██ █  ██ ██ █ ██ █ █│
        # │█  ██ █  ██ █  █  ██ █  ██ │
        # │██ █  ██  █ ██ ██  █ ██  ██│
        # │ █ ██  █ ██  █  █ ██  █ ██ │
        # │██  █ ██  ██ █ ██  ██ █  █ │
        # └────────────────────────────┘
        # 特点: 均匀分散，无空间结构，类似 MAE
        total = H * W
        target = max(1, int(round(r * total)))
        noise = torch.randn(B*V, 1, H, W, device=device)
        flat = noise.view(B*V, -1)
        kth = torch.topk(flat, target, dim=-1).values.min(dim=-1, keepdim=True).values
        mask = (flat >= kth).view(B, V, H, W)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return mask.to(dtype)


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


class G2VLMConfig(PretrainedConfig):
    def __init__(
        self,
        visual_und=True,
        visual_recon=True,
        joint_train_recon=False,
        pretrain_train_recon=False, 
        use_dinov3=False,
        ce_loss_dino=False,
        train_conf_pi3=False, 
        ssl=False,
        llm_config=None,
        vit_config=None,
        dino_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        dino_max_num_patch_per_side=37,
        interpolate_pos=False,
        use_registers=False,
        use_dino_masking=False,
        dino_mask_mode=None,
        dino_mask_ratio=None,
        dino_num_ref=-1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_und = visual_und
        self.visual_recon = visual_recon
        self.train_conf_pi3 = train_conf_pi3
        self.ssl = ssl
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.dino_config = dino_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.dino_max_num_patch_per_side = dino_max_num_patch_per_side
        self.interpolate_pos = interpolate_pos
        self.use_registers = use_registers
        self.joint_train_recon = joint_train_recon
        self.pretrain_train_recon = pretrain_train_recon
        self.use_dinov3 = use_dinov3
        self.ce_loss_dino = ce_loss_dino
        self.use_dino_masking = use_dino_masking
        self.dino_mask_mode = dino_mask_mode if dino_mask_mode is not None else ['random']
        self.dino_mask_ratio = dino_mask_ratio if dino_mask_ratio is not None else [0.5]
        self.dino_num_ref = dino_num_ref


class G2VLM(PreTrainedModel):
    config_class = G2VLMConfig
    base_model_prefix = 'g2vlm'

    def __init__(self, language_model, vit_model, dino_model, config: G2VLMConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        self.conf_head = None
        self.global_point_head = None
        self.camera_head = None
        self.point_head = None
        self.use_dinov3 = config.use_dinov3
        self.ce_loss_dino = config.ce_loss_dino

        
        if config.visual_recon:
            self.dino_model = dino_model
            self.dino_patch_size = config.dino_config.patch_size #14 
            self.dino_max_num_patch_per_side = config.dino_max_num_patch_per_side
            self.dino_hidden_size = config.dino_config.hidden_size
            self.embed_dim = self.hidden_size  
            self.resnet_normalize = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)
            self.dino2llm = nn.Linear(self.dino_hidden_size, self.hidden_size) 
            self.use_registers = config.use_registers
            self.train_conf_pi3 = config.train_conf_pi3
            self.ssl = config.ssl
            
            # Masking support for DINO tokens
            self.use_dino_masking = config.use_dino_masking
            self.dino_mask_mode = config.dino_mask_mode
            self.dino_mask_ratio = config.dino_mask_ratio
            self.num_ref = config.dino_num_ref
            if self.use_dino_masking:
                self.mask_placeholder = nn.Parameter(torch.randn(1, 1, 1, self.dino_hidden_size))
            
            if self.use_registers:
                self.register_token = nn.Parameter(torch.randn(1, 2, 4, self.hidden_size))

            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float('rope100'[len('rope'):])
            self.pi3rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()

            if self.use_registers:
                num_register_tokens = 5
                self.patch_start_idx = num_register_tokens
                self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.hidden_size))
            else: 
                self.patch_start_idx = 0 
            self.point_decoder = Pi3TransformerDecoder(
                in_dim=self.hidden_size,   #2*self.dec_embed_dim, 
                dec_embed_dim=self.hidden_size, #1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.pi3rope,
            )
            # SSL decoder
            if self.use_dinov3:
                self.ssl_decoder = SSLDecoder(patch_size=16, dec_embed_dim=self.hidden_size, output_dim=3)
                self.conf_loss = ConfLoss(patch_size=16)
            else:
                self.ssl_decoder = SSLDecoder(patch_size=14, dec_embed_dim=self.hidden_size, output_dim=3)
                self.conf_loss = ConfLoss(patch_size=14)

            if self.use_dinov3:
                self.point_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=3)
            else:
                self.point_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
            # ----------------------
            #  Camera Pose Decoder
            # ----------------------

            self.camera_decoder = Pi3TransformerDecoder(
                in_dim=self.hidden_size,
                dec_embed_dim=self.hidden_size,
                dec_num_heads=16,                
                out_dim=512,
                rope=self.pi3rope,
                use_checkpoint=False
            )
            self.camera_head = Pi3CameraHead(dim=512)
            # ----------------------
            #  Global Points Decoder
            # ----------------------
            use_global_points = True  
            self.use_global_points = use_global_points

            if use_global_points:
                self.global_points_decoder = Pi3ContextTransformerDecoder(
                    in_dim=self.hidden_size,
                    dec_embed_dim=self.hidden_size,
                    dec_num_heads=16,
                    out_dim=1024,
                    rope=self.pi3rope,
                )
                if self.use_dinov3:
                    self.global_point_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=3)
                else:
                    self.global_point_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
            else:
                self.global_point_head = None
            
            self.Pi3Loss = Pi3Loss(self.train_conf_pi3)

            if self.train_conf_pi3:
                # assert ckpt is not None

                # ----------------------
                #     Conf Decoder
                # ----------------------
                self.conf_decoder = deepcopy(self.point_decoder)
                if self.use_dinov3:
                    self.conf_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=1)
                else:
                    self.conf_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

                freeze_all_params([self.dino_model, self.dino2llm, self.language_model, self.point_decoder, self.point_head, self.camera_decoder,  self.camera_head])
                freeze_all_params([self.Pi3Loss.point_loss.segformer])
                if use_global_points:
                    freeze_all_params([self.global_points_decoder, self.global_point_head])
            else:
                self.conf_head = None 

    
  
        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = 32 
            self.vit_hidden_size = config.vit_config.hidden_size
            self.use_registers = config.use_registers
       
        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_recon:
            # weight: N(0, 0.02^2)
            # nn.init.trunc_normal_(w_fp32, mean=0.0, std=0.02, a=-0.04, b=0.04)  # ±2σ
            # bias 仍然置 0（常见做法）
            # nn.init.zeros_(self.dino2llm.bias)
            nn.init.xavier_uniform_(self.dino2llm.weight) 
            nn.init.zeros_(self.dino2llm.bias)
        if self.use_registers:
            nn.init.normal_(self.register_token, std=1e-6)
        if self.use_dino_masking:
            nn.init.trunc_normal_(self.mask_placeholder, mean=0.0, std=0.02, a=-0.04, b=0.04)

    # ----------------------
    #  Attention Map Extraction Methods
    # ----------------------
    def enable_qk_saving(self, enable: bool = True, layer_indices: list = None):
        """
        Enable Q/K state saving for attention layers to compute attention maps.
        
        Args:
            enable: Whether to enable Q/K saving
            layer_indices: List of layer indices to enable saving for. 
                          If None, enables for all layers.
        """
        for idx, layer in enumerate(self.language_model.model.layers):
            if layer_indices is None or idx in layer_indices:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'enable_qk_saving'):
                    layer.self_attn.enable_qk_saving(enable)
    
    def get_saved_attention_maps(self, layer_indices: list = None, 
                                  cu_seqlens: torch.Tensor = None,
                                  scale: float = None) -> dict:
        """
        Compute attention maps from saved Q/K states.
        
        Args:
            layer_indices: List of layer indices to get attention from.
                          If None, gets from all layers with saved states.
            cu_seqlens: Cumulative sequence lengths for variable-length sequences.
            scale: Scaling factor for attention. If None, uses 1/sqrt(head_dim).
            
        Returns:
            Dictionary mapping layer index to attention maps (head-averaged).
        """
        attention_maps = {}
        for idx, layer in enumerate(self.language_model.model.layers):
            if layer_indices is not None and idx not in layer_indices:
                continue
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'get_saved_qk_states'):
                q, k = layer.self_attn.get_saved_qk_states()
                if q is not None and k is not None:
                    # Compute attention: softmax(Q @ K^T / sqrt(d))
                    # q, k shape: [total_tokens, num_heads, head_dim]
                    head_dim = q.shape[-1]
                    if scale is None:
                        scale = 1.0 / (head_dim ** 0.5)
                    
                    if cu_seqlens is not None:
                        # Variable length sequences - compute per sequence
                        seq_attns = []
                        num_seqs = len(cu_seqlens) - 1
                        for i in range(num_seqs):
                            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                            q_seq = q[start:end]  # [seq_len, num_heads, head_dim]
                            k_seq = k[start:end]  # [seq_len, num_heads, head_dim]
                            # [num_heads, seq_len, seq_len]
                            attn = torch.einsum('qhd,khd->hqk', q_seq, k_seq) * scale
                            attn = torch.softmax(attn, dim=-1)
                            # Average over heads: [seq_len, seq_len]
                            attn = attn.mean(dim=0)
                            seq_attns.append(attn)
                        attention_maps[idx] = seq_attns
                    else:
                        # Single sequence
                        # [num_heads, total_tokens, total_tokens]
                        attn = torch.einsum('qhd,khd->hqk', q, k) * scale
                        attn = torch.softmax(attn, dim=-1)
                        # Average over heads: [total_tokens, total_tokens]
                        attn = attn.mean(dim=0)
                        attention_maps[idx] = attn
        return attention_maps
    
    def clear_saved_qk_states(self):
        """Clear all saved Q/K states to free memory."""
        for layer in self.language_model.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_saved_query_states'):
                layer.self_attn._saved_query_states = None
                layer.self_attn._saved_key_states = None

    def random_masking(self, x, B, V, Hp, Wp, mask_mode, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Masked tokens are replaced with a learnable mask_placeholder.
        x: [B, V, L, D], sequence (batch, views, spatial tokens, dim)
        Hp: patch height
        Wp: patch width
        """
        BV, L, C = x.shape
        x = x.reshape(B, V, L, C)
        # 随机选择mask模式和比例
        i = random.randint(0, len(mask_mode) - 1)
        mode, ratio = mask_mode[i], mask_ratio[i]
        # 确定参考视图的数量
        if self.num_ref >= 0:
            num_ref = self.num_ref # 固定参考视图数量
        else:
            num_ref = random.randint(V // 4, V // 2) # 随机选择 V/4 ~ V/2个参考视图

        # Generate masks only for non-reference views, 只针对非参考视图生成mask
        mask = generate_connected_masks(B, V - num_ref, Hp, Wp, ratio=ratio,
                                        device=x.device, dtype=torch.float32,
                                        mode=mode)
        mask = mask.flatten(2, 3)  # (B, V-num_ref, Hp*Wp)
        # Prepend zeros for reference views (no masking) 参考视图的 mask 全为 0
        mask = torch.cat((torch.zeros(B, num_ref, L, device=x.device), mask), dim=1) # 1代表mask 0代表没有mask

        # Replace masked tokens with learnable mask_placeholder, 用占位符替换被 mask 的 token
        x = torch.where(mask.unsqueeze(-1) > 0.5, self.mask_placeholder, x)
        x = x.reshape(B*V, L, C)
        return x, mask

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
        packed_vit_images: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        packed_image_grid_thw:  Optional[torch.IntTensor] = None,

        #reconstruct
        packed_dino_tokens: Optional[torch.Tensor] = None,
        packed_dino_token_indexes: Optional[torch.LongTensor] = None,
        packed_dino_position_ids: Optional[torch.LongTensor] = None,
        dino_token_seqlens: Optional[torch.IntTensor] = None,
        patchified_images_shapes: Optional[List[Tuple[int, int]]] = None,

        packed_dino_image_tensor_list: Optional[torch.Tensor] = None,
        packed_depths: Optional[torch.Tensor] = None,
        packed_extrinsics: Optional[torch.Tensor] = None,
        packed_intrinsics: Optional[torch.Tensor] = None,
        packed_cam_points: Optional[torch.Tensor] = None,
        packed_world_points: Optional[torch.Tensor] = None,
        packed_point_masks: Optional[torch.Tensor] = None,
        img_per_seq_lens: Optional[torch.Tensor] = None,
        query_points: Optional[torch.Tensor] = None,
        packed_view_infos=None, 
        packed_image_paths=None, 
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

        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        
        # 根据token特性准备attention mask
        if self.config.visual_recon:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:### regular mask 
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


            image_embeds = self.vit_model(packed_vit_images, grid_thw=packed_image_grid_thw)

            packed_vit_token_embed = image_embeds

            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed  

        ### visual recon分支，Geo-Multi-View Token编辑（维度变换、替换， mask）
        if self.config.visual_recon and dino_token_seqlens is not None:
            # 这段代码是将"扁平打包"的多图数据拆解为结构化的 batch 信息，同时为 DINO 模型准备 Flash Attention 所需的变长序列参数
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(dino_token_seqlens, dim=0), (1, 0)) # dino_token_seqlens：一个 1D 张量，记录每张图像经过 DINO patch 化后的 token 数量。例如 [1369, 1369, 1369] 表示 3 张图各有 1369 个 patch token（即 37×37）。 torch.cumsum(...)：对序列长度做累积求和，得到每张图的结束位置，例如 [1369, 2738, 4107]. Pad: 在最前面补一个 0，变成 [0, 1369, 2738, 4107]。这就是 cumulative sequence lengths（cu_seqlens），是 Flash Attention 的 varlen 接口要求的格式，用来告诉 attention 哪些 token 属于同一张图（避免跨图 attention）
            cu_seqlens = cu_seqlens.to(torch.int32) 
            max_seqlen = torch.max(dino_token_seqlens).item() # 所有图像中最大的 token 序列长度，也是 Flash Attention 的必要参数
      

            BS, C_in, H, W = packed_dino_image_tensor_list.shape # 其中 BS 是所有图像打包在一起的总数（batch × 每个样本的图像数）
        
            S = img_per_seq_lens[0] #constant for now 每个样本（序列）包含的图像数量。注释 #constant for now 说明当前假设每个样本的图像数相同。
            B = BS // S # 反推出 batch size。例如 BS=6, S=3 → B=2，即 2 个样本，每个样本 3 张图
            if self.use_dinov3:
                patch_h, patch_w = H // 16, W // 16
            else:
                patch_h, patch_w = H // 14, W // 14 # DINOv2 将图像划分为 14×14 像素的小块（patch），每个 patch 被编码为一个 token, 例如一张 518×518 的图像：518 // 14 = 37，得到 37×37 = 1369 个 patch token
            if self.config.joint_train_recon or self.config.pretrain_train_recon:
                images = packed_dino_image_tensor_list.reshape(B, S, C_in, H, W)

                ## undo resnet_norm DINO 需要标准化后的输入来提取特征，但损失函数(pi3-loss)和可视化需要原始像素值，所以在把图像传给 loss 之前要 undo 这个标准化
                resmean = torch.tensor(_RESNET_MEAN).view(1, 1, -1, 1, 1).to(images.device)
                resstd = torch.tensor(_RESNET_STD).view(1, 1, -1, 1, 1).to(images.device)
                images_unorm = images * resstd + resmean 
        
                batch = {}
                batch['depths'] = packed_depths.reshape(B, S, *packed_depths.shape[1:]).to(torch.float32)
                batch['extrinsics'] = packed_extrinsics.reshape(B, S, *packed_extrinsics.shape[1:]).to(torch.float32)
                batch['intrinsics'] = packed_intrinsics.reshape(B, S, *packed_intrinsics.shape[1:]).to(torch.float32)
                # batch['cam_points'] = packed_cam_points.reshape(B, S, *packed_cam_points.shape[1:])
                batch['world_points'] = packed_world_points.reshape(B, S, *packed_world_points.shape[1:]).to(torch.float32)
                batch['point_masks'] = packed_point_masks.reshape(B, S, *packed_point_masks.shape[1:])
                batch['images'] = images_unorm
                batch['view_infos'] = packed_view_infos
                batch['image_paths'] = packed_image_paths
     
                if self.use_dinov3:
                    packed_dino_token_embed = self.dino_model(
                        # packed_pixel_values=packed_dino_image_tensor_list, 
                        pixel_values=packed_dino_image_tensor_list,
                        # packed_flattened_position_ids=packed_vit_position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
                else:
                    packed_dino_token_embed = self.dino_model(
                        packed_pixel_values=packed_dino_image_tensor_list, 
                        # packed_flattened_position_ids=packed_vit_position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )


                BS, P, D = packed_dino_token_embed.size() #
                if self.ssl:
                    packed_dino_token_embed, mask = self.random_masking(packed_dino_token_embed, B, S, patch_h, patch_w, mask_mode=self.dino_mask_mode, mask_ratio=self.dino_mask_ratio)
                    packed_dino_token_embed = packed_dino_token_embed.reshape(BS*P, D)
                packed_dino_token_embed = self.dino2llm(packed_dino_token_embed) # 768 -> 1536
                _, D = packed_dino_token_embed.shape
                packed_dino_token_embed = packed_dino_token_embed.reshape(BS, -1, D) 
            
                if self.use_registers:
                    # 每张图后面都加上registers
                    register_token = self.register_token.repeat(B, S, 1, 1).reshape(B*S, *self.register_token.shape[-2:])
                    packed_dino_token_embed = torch.cat([register_token, packed_dino_token_embed], dim=1)

                packed_dino_token_embed = packed_dino_token_embed.reshape(-1, D)

                packed_sequence[packed_dino_token_indexes] = packed_dino_token_embed
            
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_geo_token_indexes=packed_dino_token_indexes, 
            )

        if self.config.visual_recon and dino_token_seqlens is not None:

            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                output_hidden_states=False,
                **extra_inputs,
            )
            selected_hidden_states = last_hidden_state.hidden_states
            last_hidden_state = last_hidden_state.packed_query_sequence
        else: 
            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                output_hidden_states=False,
                **extra_inputs,
            )
            last_hidden_state = last_hidden_state.packed_query_sequence


        vggt_loss_dict = None
        dl_loss = 0 
        details = {}
        predictions = {}
        ssl_loss = None
        # loss 分支 recon
        if self.config.visual_recon:
 
            if self.config.joint_train_recon or self.config.pretrain_train_recon:
                # 提取 DINO token 对应的隐藏状态
                N = S # 视图数量
                hidden = last_hidden_state[packed_dino_token_indexes].reshape(B*S, -1, D)
                hw = hidden.shape[1]
                if self.ssl:
                    # 去掉register token, 因为不对应任何图像 patch，它们没有空间位置，无法解码成像素
                    if self.use_registers:
                        x = hidden[:, self.patch_start_idx:]
                    else:
                        x = hidden
                    pred, conf = self.ssl_decoder(x) # LayerNorm -> Linear(E → p²×3×2) -> 分出预测和置信度
                    conf = torch.sigmoid(conf)
                    # reshape from (B*S, L, p²×3) to (B, S, L, p²×3) to match ConfLoss expectation
                    pred = pred.reshape(B, S, *pred.shape[1:])
                    conf = conf.reshape(B, S, *conf.shape[1:])
                    ssl_loss, details = self.conf_loss(images_unorm, pred, conf, mask)
                    predictions["images"] = images_unorm
                    predictions["mask"] = mask
                    predictions["pred"] = pred
                    predictions["conf"] = conf
                else:
                    # 准备GT数据
                    predictions['world_points'] = batch['world_points']
                    predictions['point_masks'] = batch['point_masks']
                    predictions['view_infos'] = batch['view_infos']
                    predictions['image_paths'] = batch['image_paths']
                    # 构造 2D 位置编码（RoPE 用）供后续 Transformer 解码器中的 RoPE2D 旋转位置编码使用
                    if self.use_dinov3:
                        pos = self.position_getter(B * N, H//16, W//16, hidden.device)
                    else:
                        pos = self.position_getter(B * N, H//14, W//14, hidden.device)
                    if self.patch_start_idx > 0:
                    
                        pos = pos + 1
                        pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
                        pos = torch.cat([pos_special, pos], dim=1)
                
                    pos = pos.reshape(B*N, hw, -1)
                    # 四个解码器并行工作
                    # point_decoder	point_hidden	解码局部3D点（相机坐标系下）
                    # conf_decoder	conf_hidden	解码置信度（可选，仅 train_conf_pi3 时）
                    # camera_decoder	camera_hidden	解码相机位姿
                    # global_points_decoder	global_point_hidden	解码全局3D点（世界坐标系下）
                    # 其中 global_points_decoder 是一个 交叉注意力解码器，它用第一个视图的特征作为 context（参考帧），帮助其他视图对齐到全局坐标系
                    point_hidden = self.point_decoder(hidden, xpos=pos)
                    if self.train_conf_pi3:
                        conf_hidden = self.conf_decoder(hidden, xpos=pos)
                    camera_hidden = self.camera_decoder(hidden, xpos=pos)
                    if self.use_global_points:
                        context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
                        global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)
                    
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        # local points
                        point_hidden = point_hidden.float()
                        ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                        xy, z = ret.split([2, 1], dim=-1)
                        z = torch.exp(z)
                        local_points = torch.cat([xy * z, z], dim=-1)

                        # confidence
                        if self.train_conf_pi3:
                            conf_hidden = conf_hidden.float()
                            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                        else:
                            conf = None
                            
                        # camera
                        camera_hidden = camera_hidden.float()
                        camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

                        # Global points
                        if self.use_global_points:
                            global_point_hidden = global_point_hidden.float()
                            global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                        else:
                            global_points = None
                        # unproject local points using camera poses
                        points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
                    
                    pi3_pred = dict(
                        points=points,
                        local_points=local_points,
                        conf=conf,
                        camera_poses=camera_poses,
                        global_points=global_points
                    )
                    predictions['points'] = points
                    predictions['camera_poses'] = camera_poses
                    predictions['local_points'] = local_points
                    predictions['global_points'] = global_points

                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16): 
                        dl_loss, details = self.Pi3Loss(pi3_pred, batch)


                    predictions["images"] = images_unorm
        # loss 分支 language token ce
        ce = None
        mse = None # 用于visual gen
        if ce_loss_indexes is not None:
    
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])

            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none") #note because here it employed 

        
        if vggt_loss_dict is not None:
            if 'loss_reg_point' in vggt_loss_dict: 
                return dict(mse=mse, ce=ce, dl=vggt_loss, depth_loss_reg=vggt_depth_loss_reg, \
                            depth_loss_conf=vggt_depth_loss_conf, point_loss_reg=vggt_point_loss_reg, \
                            camera_loss=vggt_camera_loss, point_loss_conf=vggt_point_loss_conf, \
                            camera_auc_30=camera_auc_30, camera_auc_20=camera_auc_20, camera_auc_10=camera_auc_10,\
                                camera_auc_5=camera_auc_5,camera_auc_3=camera_auc_3), predictions
            else: 
                return dict(mse=mse, ce=ce, dl=vggt_loss, depth_loss_reg=vggt_depth_loss_reg, \
                            depth_loss_conf=vggt_depth_loss_conf, \
                            camera_loss=vggt_camera_loss, \
                            camera_auc_30=camera_auc_30, camera_auc_20=camera_auc_20, camera_auc_10=camera_auc_10,\
                                camera_auc_5=camera_auc_5,camera_auc_3=camera_auc_3), predictions
        else: 
    
            return  EasyDict(
                # mse=mse, 
                ce=ce,
                ssl_loss=ssl_loss,
                # dl=dl_loss,
                **details
            ), predictions 


    def prepare_prompts_addbos(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
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
            text_ids = [new_token_ids['bos_token_id']] + text_ids 
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
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_prompts_addeos(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
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
            assistant_ids = tokenizer.encode('assistant\n')
            text_ids = text_ids + [new_token_ids['eos_token_id']] + [new_token_ids['bos_token_id']] + assistant_ids
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
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
    def prepare_prompts_pure_text(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
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
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
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
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
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
        packed_vit_images = list()
        packed_image_grid_thw = list()
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
         
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            image_tensor,  image_grid_thw = transforms([image])
            packed_image_grid_thw.append(image_grid_thw[0])
            num_img_tokens = image_tensor.shape[0] // 4 
            packed_vit_images.append(image_tensor)
            
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens


            postions_ids_from_vit_for_rope, rope_deltas = get_rope_index_image_3D(
                image_grid_thw[0],
                curr_position_id,
                device=image_tensor.device
            )

            packed_position_ids.extend([postions_ids_from_vit_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1


            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_image_grid_thw": torch.stack(packed_image_grid_thw, dim=0),
            "packed_vit_images": torch.stack(packed_vit_images, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
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
        packed_vit_images: torch.Tensor,
        packed_image_grid_thw:  torch.IntTensor,
        packed_vit_token_indexes: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_vit_tokens: Optional[torch.Tensor]=None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
    ):  

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()

        image_embeds = self.vit_model(packed_vit_images, grid_thw=packed_image_grid_thw)
        packed_vit_token_embed = image_embeds



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
    
    def prepare_dino_images_pi3 (self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_dino_token_indexes = list()
        dino_token_seqlens, packed_dino_tokens, packed_dino_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
    
        vggt_fixed_resolution = 518 # hardcode 
        img_load_resolution = 1024

        images = load_and_resize14(images,vggt_fixed_resolution)

        curr_kvlen = curr_kvlens[0]
        curr_position_id = curr_rope[0]
        
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen
        for image in images:
            
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = image
            height, width = image_tensor.shape[1:]
            grid_t = 1  
            grid_h, grid_w = height // 14, width // 14

            # add 3d pos for <|startofimage|> token
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            dino_tokens = patchify(image_tensor, self.dino_patch_size)
            packed_dino_tokens.append(dino_tokens)# 实际没有使用这个
            num_img_tokens = dino_tokens.shape[0]
            dino_token_seqlens.append(num_img_tokens)
  
            packed_dino_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            ###3d rope embedding for QKV attention: 
            dino_image_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) 
            postions_ids_from_dino_for_rope, rope_deltas = get_rope_index_image_3D_dino(
                dino_image_thw,
                curr_position_id,
                device=image_tensor.device
            )
            packed_position_ids.extend([postions_ids_from_dino_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            curr_kvlen += num_img_tokens + 2

            new_rope.append(curr_position_id)

        newlens = [newlens[-1]]
        new_rope = [new_rope[-1]]
        packed_seqlens = [sum(packed_seqlens)]


        assert len(images.shape) == 4
        assert images.shape[1] == 3
        original_images = images.clone()
        images = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)(images) 

        generation_input = {
            "packed_dino_images": images, 
            'original_images': original_images,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "dino_token_seqlens": torch.tensor(dino_token_seqlens, dtype=torch.int),
            "packed_dino_token_indexes": torch.tensor(packed_dino_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
    
        return generation_input, newlens, new_rope
    def prepare_dino_images_none (self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_dino_token_indexes = list()
        dino_token_seqlens, packed_dino_tokens, packed_dino_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
    
        vggt_fixed_resolution = 518 # hardcode 
        img_load_resolution = 1024

        curr_kvlen = curr_kvlens[0]
        curr_position_id = curr_rope[0]
        
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen
        for image in images:
            
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = image
            height, width = image_tensor.shape[1:]
            grid_t = 1  
            grid_h, grid_w = height // 14, width // 14

            # add 3d pos for <|startofimage|> token
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            dino_tokens = patchify(image_tensor, self.dino_patch_size)
            packed_dino_tokens.append(dino_tokens)
            num_img_tokens = dino_tokens.shape[0]
            dino_token_seqlens.append(num_img_tokens)
  
            packed_dino_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            ###3d rope embedding for QKV attention: 
            dino_image_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) 
            postions_ids_from_dino_for_rope, rope_deltas = get_rope_index_image_3D_dino(
                dino_image_thw,
                curr_position_id,
                device=image_tensor.device
            )
            packed_position_ids.extend([postions_ids_from_dino_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            curr_kvlen += num_img_tokens + 2

            new_rope.append(curr_position_id)

        newlens = [newlens[-1]]
        new_rope = [new_rope[-1]]
        packed_seqlens = [sum(packed_seqlens)]


        assert len(images.shape) == 4
        assert images.shape[1] == 3
        original_images = images.clone()
        images = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)(images) 

        generation_input = {
            "packed_dino_images": images, 
            'original_images': original_images,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "dino_token_seqlens": torch.tensor(dino_token_seqlens, dtype=torch.int),
            "packed_dino_token_indexes": torch.tensor(packed_dino_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
    
        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_dino(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_dino_token_indexes: torch.LongTensor,
        dino_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_dino_images: torch.Tensor, 
        original_images: torch.Tensor, 
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(dino_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(dino_token_seqlens).item()
 
        packed_dino_token_embed = self.dino_model(
            packed_pixel_values=packed_dino_images, 
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        B, P, D = packed_dino_token_embed.size() #
        packed_dino_token_embed = packed_dino_token_embed.reshape(B*P, D)

        packed_dino_token_embed = self.dino2llm(packed_dino_token_embed)

        BS, C_in, H, W = packed_dino_images.shape
        S = BS ### constant for now 
        B = BS // S 
        assert B==1

        if packed_dino_token_embed.dtype != packed_sequence.dtype:
            packed_dino_token_embed = packed_dino_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_dino_token_indexes] = packed_dino_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "geo",
                "packed_geo_token_indexes": packed_dino_token_indexes, 
                "packed_text_indexes": packed_text_indexes
            } 

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            output_hidden_states=False, 
            is_causal=False,
            **extra_inputs,
        )

     
        past_key_values = output.past_key_values
        last_hidden_state = output.packed_query_sequence


        return past_key_values, last_hidden_state


    def prepare_start_tokens(self, curr_kvlens, curr_rope, tokenizer, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        template = "<|im_start|>user\your text<|im_end|>\n<|im_start|>assistant\n"
        
        template_ids = tokenizer.encode(template, add_special_tokens=False)
        if template_ids:
            start_token_id = template_ids[-1]  
        else:
            start_token_id = tokenizer.eos_token_id or 151643 

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(start_token_id)
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long).expand(3, -1),
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

    @torch.no_grad
    def reconstruct(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        selected_hidden_states: torch.Tensor, 
        packed_dino_token_indexes: torch.Tensor,
        packed_dino_images: torch.Tensor,
        original_images: torch.Tensor,
        **kwargs,
    ):
        ps_idx = 0  # hardcode 
        BS, C_in, H, W = packed_dino_images.shape
        B = 1
        S = BS // B
        N = S
        if len(original_images.shape) == 4:
            original_images = original_images.unsqueeze(0)
        dino_images=packed_dino_images[None]
        print('original_images', original_images.shape)
        print('dino_images', dino_images.shape)

        aggregated_tokens_list = []
        _, D = selected_hidden_states.shape ### this is actually only last hidden 
        hidden = selected_hidden_states[packed_dino_token_indexes].reshape(B*S, -1, D)

    
        hw = hidden.shape[1]
        if self.use_dinov3:
            pos = self.position_getter(B * N, H//16, W//16, hidden.device)
            patch_h, patch_w = H // 16, W // 16
        else:
            pos = self.position_getter(B * N, H//14, W//14, hidden.device)
            patch_h, patch_w = H // 14, W // 14

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        pos = pos.reshape(B*N, hw, -1)

        ### return original images
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            point_hidden = self.point_decoder(hidden, xpos=pos)
            if self.conf_head is not None:
                conf_hidden = self.conf_decoder(hidden, xpos=pos)
            camera_hidden = self.camera_decoder(hidden, xpos=pos)
            if self.use_global_points:
                context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
                global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)
            
            # local points
            with torch.amp.autocast(device_type='cuda', enabled=False):
                point_hidden = point_hidden.float()
                ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                xy, z = ret.split([2, 1], dim=-1)
                z = torch.exp(z)
                local_points = torch.cat([xy * z, z], dim=-1)

                # confidence
                if self.conf_head is not None:
                    conf_hidden = conf_hidden.float()
                    conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                else:
                    conf = None
                    
                # camera
                camera_hidden = camera_hidden.float()
                camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

                # Global points
                if self.use_global_points:
                    global_point_hidden = global_point_hidden.float()
                    global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                else:
                    global_points = None
                
                # unproject local points using camera poses
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        pi3_pred = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
            
        )
        pi3_pred['images'] = original_images
    
        return pi3_pred
  
    @torch.no_grad()
    def recon(
        self,
        tokenizer,
        new_token_ids,
        dino_image_transform,
        images, #this now expect image paths 
        prompt='Reconstruct the 3D scene.',
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

        # system_prompt = 'system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n'
        system_prompt = 'Reconstruct the 3D scene.'

        print('Prepareing prompt')
        generation_input, newlens, new_rope = self.prepare_prompts_addbos(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[system_prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        print('Prepareing dino images ')

        # generation_input, newlens, new_rope = self.prepare_dino_images_none(
        generation_input, newlens, new_rope = self.prepare_dino_images_pi3(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=images, 
            transforms=dino_image_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values, last_hidden_state = self.forward_cache_update_dino(past_key_values, **generation_input)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            predictions_dict = self.reconstruct(
                past_key_values=past_key_values,
                selected_hidden_states=last_hidden_state,
                **generation_input,
            )

        return predictions_dict
    @torch.no_grad()
    def recon_for_eval(
        self,
        tokenizer,
        new_token_ids,
        dino_image_transform,
        images, #this now expect image paths 
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

        #add text system prompt hard code: 
        # system_prompt = 'system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n'
        system_prompt = 'Reconstruct the 3D scene.'
        ### todo check what is this template, does it needs bos eos .... 

        print('Prepareing prompt')
        generation_input, newlens, new_rope = self.prepare_prompts_addbos(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[system_prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # add images
        # for image in images:
        print('Prepareing dino images ')

        generation_input, newlens, new_rope = self.prepare_dino_images_none(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=images, 
            transforms=dino_image_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values, last_hidden_state = self.forward_cache_update_dino(past_key_values, **generation_input)

        # recon 
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            predictions_dict = self.reconstruct(
                past_key_values=past_key_values,
                max_length=max_length,
                selected_hidden_states=last_hidden_state,
                # do_sample=do_sample,
                # temperature=temperature,
                # end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )

        return predictions_dict

    @torch.no_grad()
    def chat_with_recon(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        dino_image_transform,
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

        #add text system prompt hard code: 
        system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        generation_input, newlens, new_rope = self.prepare_prompts_pure_text(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[system_prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
            
        generation_input, newlens, new_rope = self.prepare_dino_images_pi3(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=images.copy(), 
            transforms=dino_image_transform,
            new_token_ids=new_token_ids,
        )
        tmp_save = {}
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
                tmp_save[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values, last_hidden_state = self.forward_cache_update_dino(past_key_values, **generation_input)
            
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
        prompt = prompt+'<|im_end|>\n<|im_start|>assistant'
        generation_input, newlens, new_rope = self.prepare_prompts_pure_text( #self.prepare_prompts_addeos(
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
        generation_input = self.prepare_start_tokens(newlens, new_rope,tokenizer, new_token_ids)
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
        
        # skip the start token
        output = tokenizer.decode(unpacked_latent[1:,0])
        return output

    def _soft_argmax(self, corr, x_normal, y_normal, beta=0.02):
        """
        Soft argmax for computing sub-pixel accurate coordinates from correlation map.
        Reference: SFNet: Learning Object-aware Semantic Flow (Lee et al.)
        
        Args:
            corr: (B, h*w, h, w) correlation map, where h*w is source pixels
            x_normal: (w,) normalized x coordinates [-1, 1]
            y_normal: (h,) normalized y coordinates [-1, 1]
            beta: temperature for softmax (smaller = sharper)
            
        Returns:
            grid_x: (B, 1, h, w) x coordinates for each source pixel
            grid_y: (B, 1, h, w) y coordinates for each source pixel
        """
        b, _, h, w = corr.size()
        
        # Apply softmax with temperature along the target dimension
        corr = F.softmax(corr / beta, dim=1)  # (B, h*w, h, w)
        corr = corr.view(b, h, w, h, w)  # (B, h_src, w_src, h_tgt, w_tgt)
        
        # Compute expected x coordinate
        # Sum over h_tgt (dim=3) to get distribution over w_tgt
        grid_x = corr.sum(dim=3, keepdim=False)  # (B, h_src, w_src, w_tgt)
        x_normal = x_normal.expand(b, w)  # (B, w)
        x_normal = x_normal.view(b, 1, 1, w)  # (B, 1, 1, w_tgt)
        grid_x = (grid_x * x_normal).sum(dim=3, keepdim=True)  # (B, h_src, w_src, 1)
        grid_x = grid_x.permute(0, 3, 1, 2)  # (B, 1, h_src, w_src)
        
        # Compute expected y coordinate
        # Sum over w_tgt (dim=4) to get distribution over h_tgt
        grid_y = corr.sum(dim=4, keepdim=False)  # (B, h_src, w_src, h_tgt)
        y_normal = y_normal.expand(b, h)  # (B, h)
        y_normal = y_normal.view(b, 1, 1, h)  # (B, 1, 1, h_tgt)
        grid_y = (grid_y * y_normal).sum(dim=3, keepdim=True)  # (B, h_src, w_src, 1)
        grid_y = grid_y.permute(0, 3, 1, 2)  # (B, 1, h_src, w_src)
        
        return grid_x, grid_y

    def _unnormalise_and_convert_mapping_to_flow(self, map_coords, H, W):
        """
        Convert normalized mapping [-1, 1] to pixel-level flow.
        
        Args:
            map_coords: (B, 2, H, W) normalized coordinates in [-1, 1]
            H, W: target height and width (used for reference, actual size from map_coords)
            
        Returns:
            flow: (B, 2, H, W) pixel-level flow
        """
        B, C, h, w = map_coords.size()
        device = map_coords.device
        
        # Convert normalized coords [-1, 1] to pixel coords [0, W-1] and [0, H-1]
        mapping = torch.zeros_like(map_coords)
        mapping[:, 0, :, :] = (map_coords[:, 0, :, :].float().clone() + 1) * (w - 1) / 2.0  # x
        mapping[:, 1, :, :] = (map_coords[:, 1, :, :].float().clone() + 1) * (h - 1) / 2.0  # y
        
        # Create identity grid in pixel coordinates
        xx = torch.arange(0, w, device=device).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h, device=device).view(-1, 1).repeat(1, w)
        xx = xx.view(1, 1, h, w).repeat(B, 1, 1, 1).float()
        yy = yy.view(1, 1, h, w).repeat(B, 1, 1, 1).float()
        grid = torch.cat((xx, yy), 1)  # (B, 2, h, w)
        
        # Flow = mapping - identity grid
        flow = mapping - grid
        
        return flow

    def _get_cross_view_correlation(
        self,
        attn_list: list,
        src_start: int,
        src_end: int,
        tgt_start: int,
        tgt_end: int,
    ):
        """
        从多层 attention maps 中提取跨视图相关性并融合。
        
        Args:
            attn_list: list of attention tensors, 每个 shape 为 [total_tokens, total_tokens] (已对 heads 平均)
            src_start, src_end: 源视图 token 的起止索引
            tgt_start, tgt_end: 目标视图 token 的起止索引
            
        Returns:
            refined_corr: (L_src, L_tgt) 融合后的跨视图相关性矩阵
        """
        sim_s_to_t_list = []
        sim_t_to_s_list = []
        
        for attn in attn_list:
            # 提取跨视图注意力 (已经对 heads 平均过了)
            sim_s_to_t_list.append(attn[src_start:src_end, tgt_start:tgt_end])  # (L_src, L_tgt)
            sim_t_to_s_list.append(attn[tgt_start:tgt_end, src_start:src_end])  # (L_tgt, L_src)
        
        # 对所有层取平均
        sim_s_to_t = torch.stack(sim_s_to_t_list, dim=0).mean(dim=0)  # (L_src, L_tgt)
        sim_t_to_s = torch.stack(sim_t_to_s_list, dim=0).mean(dim=0)  # (L_tgt, L_src)
        
        # 双向融合
        refined_corr = (sim_s_to_t + sim_t_to_s.T) / 2
        
        return refined_corr

    @torch.no_grad
    def track_points_from_flow(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        selected_hidden_states: torch.Tensor, 
        packed_dino_token_indexes: torch.Tensor,
        packed_dino_images: torch.Tensor,
        original_images: torch.Tensor,
        start_points: torch.Tensor,  # (num_tracks, 2) 起始点坐标 (x, y)，在 output_hw 分辨率下
        source_view_idx: int = 0,    # 起始视图索引
        beta: float = 0.0001,        # soft argmax temperature
        attn_maps: dict = None,      # attention maps 字典 {layer_idx: attention_tensor}
        output_hw: tuple = None,     # (H_out, W_out) gt 分辨率；None 则使用 dino 分辨率
        **kwargs,
    ):
        """
        基于 attention maps 计算光流并追踪点。
        使用双向注意力融合和 soft argmax 来计算更准确的对应关系。
        
        Args:
            start_points: (num_tracks, 2) 在 source_view_idx 视图中的起始点坐标 (x, y)，
                          坐标系为 output_hw（即 gt 分辨率）
            source_view_idx: 起始视图的索引，默认为 0
            beta: soft argmax 的温度参数，越小越尖锐
            attn_maps: 预计算 attention maps 字典 {layer_idx: attention_tensor}
            output_hw: (H_out, W_out) 输出坐标所在的分辨率（gt 分辨率）；
                       None 则退化为 dino resize 后的分辨率
            
        Returns:
            dict containing:
                - pred_tracks: (num_tracks, V, 2) 所有视图中的追踪点坐标，坐标系为 output_hw
                - flow_maps: (V-1, 2, H_out, W_out) 从源视图到其他视图的光流图
        """
        assert attn_maps is not None, "attn_maps must be provided"
        
        BS, C_in, H, W = packed_dino_images.shape
        B = 1
        S = BS // B
        V = S  # 视图数量
        num_tracks = start_points.shape[0]
        device = packed_dino_images.device
        
        # 计算 patch 尺寸
        if self.use_dinov3:
            patch_h, patch_w = H // 16, W // 16
        else:
            patch_h, patch_w = H // 14, W // 14
        
        # 初始化追踪结果
        pred_tracks = torch.zeros((num_tracks, V, 2), device=device)
        pred_tracks[:, source_view_idx, :] = start_points.to(device)
        
        flow_maps = []
        
        # 归一化坐标网格 [-1, 1]
        x_normal = torch.linspace(-1, 1, patch_w, device=device)
        y_normal = torch.linspace(-1, 1, patch_h, device=device)
        
        # 预先提取 attention maps (支持单层或多层)
        # attn_maps 格式: {layer_idx: attn_tensor} 或 {layer_idx: [attn_per_seq]}
        attn_list = []
        for layer_idx in attn_maps.keys():
            attn = attn_maps[layer_idx]
            if isinstance(attn, list):
                attn = attn[0]  # 取第一个序列 (batch=1)
            attn_list.append(attn)  # 每个 attn shape: [total_tokens, total_tokens] (已对 heads 平均)
        
        # 每个视图有 (patch_start_idx + patch_h * patch_w) 个 token
        tokens_per_view = self.patch_start_idx + patch_h * patch_w
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for v_idx in range(V):
                if v_idx == source_view_idx:
                    continue
                
                # 计算源和目标视图的范围 (跳过 register tokens)
                src_start = source_view_idx * tokens_per_view + self.patch_start_idx
                src_end = src_start + patch_h * patch_w
                tgt_start = v_idx * tokens_per_view + self.patch_start_idx
                tgt_end = tgt_start + patch_h * patch_w
                
                # 提取跨视图相关性 (多层融合)
                refined_corr = self._get_cross_view_correlation(
                    attn_list, src_start, src_end, tgt_start, tgt_end
                )
                
                # 重塑为 (B, h*w, h, w) 用于 soft argmax
                refined_corr = refined_corr.view(1, patch_h * patch_w, patch_h, patch_w)
                
                # 使用 soft argmax 获取亚像素精度的坐标
                # grid_x, grid_y: (B, 1, h, w) - 每个源像素对应的目标坐标
                grid_x, grid_y = self._soft_argmax(refined_corr, x_normal, y_normal, beta=beta)
                
                # 组合成 mapping: (B, 2, patch_h, patch_w)
                coarse_mapping = torch.cat([grid_x, grid_y], dim=1)  # (1, 2, patch_h, patch_w)
                
                # 转换为光流
                flow_coarse = self._unnormalise_and_convert_mapping_to_flow(coarse_mapping, patch_h, patch_w)
                
                # 上采样到输出分辨率（若 output_hw 指定则用它，否则用 dino 分辨率）
                out_h, out_w = output_hw if output_hw is not None else (H, W)
                flow_full = F.interpolate(flow_coarse, size=(out_h, out_w), mode='bilinear', align_corners=False)
                
                # 缩放光流值到像素坐标
                flow_full[:, 0, :, :] *= out_w / patch_w  # x 方向
                flow_full[:, 1, :, :] *= out_h / patch_h  # y 方向
                
                flow_full = flow_full.squeeze(0)  # (2, out_h, out_w)
                flow_maps.append(flow_full)
                
                # 使用 grid_sample 采样起始点对应的光流
                # start_points 坐标系为 output_hw，flow_full 也是 output_hw 分辨率
                norm_u = 2.0 * start_points[:, 0] / (out_w - 1) - 1.0
                norm_v = 2.0 * start_points[:, 1] / (out_h - 1) - 1.0
                grid = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2).to(device)
                
                flow_vectors = F.grid_sample(
                    flow_full.unsqueeze(0),  # (1, 2, H, W)
                    grid, 
                    align_corners=False,
                    mode='bilinear',
                    padding_mode='border'
                ).squeeze()  # (2, num_tracks)
                
                if flow_vectors.dim() == 1:
                    flow_vectors = flow_vectors.unsqueeze(1)
                flow_vectors = flow_vectors.T  # (num_tracks, 2)
                
                # 计算目标视图中的点坐标
                pred_tracks[:, v_idx, :] = start_points.to(device) + flow_vectors
        
        # Stack flow maps
        if flow_maps:
            flow_maps = torch.stack(flow_maps, dim=0)  # (V-1, 2, H, W)
        else:
            flow_maps = None
        
        result = {
            'pred_tracks': pred_tracks,  # (num_tracks, V, 2)
            'flow_maps': flow_maps,      # (V-1, 2, H, W)
            'images': original_images,
        }
        
        return result
        
    @torch.no_grad()
    def track_points(
        self,
        tokenizer,
        new_token_ids,
        dino_image_transform,
        images,
        start_points: torch.Tensor,  # (num_tracks, 2) 起始点坐标，坐标系为 output_hw（gt 分辨率）
        source_view_idx: int = 0,
        beta: float = 0.0001,  # soft argmax temperature
        use_attention_maps: bool = False,  # 是否使用真实 attention maps
        attn_layer_idx: int = -1,  # 使用哪层的 attention (-1 表示最后一层)
        output_hw: tuple = None,  # (H_gt, W_gt)，gt 分辨率；None 则使用 dino resize 后的分辨率
        # prompt='Track points across views.',
    ):
        """
        完整的点追踪推理接口。
        
        Args:
            images: 图像路径列表
            start_points: (num_tracks, 2) 在第一帧中的起始点坐标 (x, y)，坐标系为 output_hw（gt 分辨率）
            source_view_idx: 起始视图索引
            beta: soft argmax 温度参数，越小越尖锐
            use_attention_maps: 是否使用真实 attention maps 而非 hidden state similarity
            attn_layer_idx: 使用哪层的 attention (-1 表示最后一层)
            output_hw: (H_gt, W_gt) 输出坐标所在的分辨率（gt 分辨率）；None 则退化为 dino 分辨率
            
        Returns:
            dict with pred_tracks (坐标系为 output_hw), flow_maps, images
        """
        device = next(self.parameters()).device
        num_layers = self.config.llm_config.num_hidden_layers
        
        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)
        
        # prefill
        past_key_values = NaiveCache(num_layers)
        newlens = [0]
        new_rope = [0]
        
        system_prompt = 'Reconstruct the 3D scene.'
        
        print('Preparing prompt')
        generation_input, newlens, new_rope = self.prepare_prompts_addbos(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[system_prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
        
        print('Preparing dino images')
        generation_input, newlens, new_rope = self.prepare_dino_images_pi3(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=images, 
            transforms=dino_image_transform,
            new_token_ids=new_token_ids,
        )
        images_no_norm = generation_input['original_images']
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        
        # Enable Q/K saving if using attention maps
        attn_maps = None
        if use_attention_maps:
            # Save Q/K for all layers so we can aggregate across them later
            self.enable_qk_saving(True, layer_indices=None)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values, last_hidden_state = self.forward_cache_update_dino(past_key_values, **generation_input)
        
        # Get attention maps if enabled (fetch all layers so we can average across them)
        if use_attention_maps:
            attn_maps = self.get_saved_attention_maps(layer_indices=None)
            self.enable_qk_saving(False)
            self.clear_saved_qk_states()
        
        # attn_maps: 每层的跨视图注意力，List of (1, V, L, V, L)
        print('Tracking points')
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tracking_result = self.track_points_from_flow(
                past_key_values=past_key_values,
                selected_hidden_states=last_hidden_state,
                start_points=start_points,
                source_view_idx=source_view_idx,
                beta=beta,
                attn_maps=attn_maps,
                output_hw=output_hw,
                **generation_input,
            )
        
        return tracking_result, images_no_norm

 