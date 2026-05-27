#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE Prefix 模式的多视角生成评估脚本（VideoLLM3D / ScanNet 数据集）。

behind_vae=False + use_partial_noise=True + no_ce_loss=True

训推一致（token-level masked reconstruction）：
  - 使用 VideoProcessor 进行帧采样（与训练时 videollm3d_dataset.py 一致）
  - VIT transform: stride=14, max_image_size=384, min_image_size=256
  - VAE transform: stride=16, max_image_size=256, min_image_size=256
  - 文本模板与训练时 apply_template_qwenvl2 一致
  - ref/nonref 分割逻辑与训练时一致
  - 掩码重建策略与训练一致：
    * ref 帧 VAE latent 保持 clean（timestep=0），写入 KV cache
    * nonref 帧 VAE latent 中被 mask 的 token 从纯噪声 denoise，未被 mask 的保持 clean
    * VIT 所有帧正常注入（无 masking）
    * mask_mode / mask_ratio 与训练脚本一致

推理流程（训推一致的 VAE prefix masked reconstruction）：
  1. 注入 VAE ref 帧的 clean latent（timestep=0）到 KV cache
  2. 注入 VIT 所有帧（ref + nonref）到 KV cache（无 masking）
  3. 注入文本 token（question + answer）到 KV cache
  4. 对所有 nonref 帧一起执行 flow-matching denoise：
     a. 被 mask 的 token 从纯噪声开始 denoise，未被 mask 的保持 clean
     b. 每个 denoise step 后，将未被 mask 的 token 重置为 clean latent
  5. 解码 VAE latent 为图片

用法:
    torchrun --nproc_per_node=8 eval/gen/gen_images_prefix_vae_videollm3d.py \
        --model-path /path/to/model \
        --output_dir ./output/prefix_vae_videollm3d \
        --dataset scanqa \
        --num_samples 100
"""

import os
import math
import copy
import json
import random
import argparse
import traceback

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file

from data.data_utils import add_special_tokens, pil_img2rgb, patchify
from data.transforms import ImageTransform
from data.video_utils import VideoProcessor
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache


# ============================================================
# VideoLLM3D 数据路径配置
# ============================================================
VIDEOLLM3D_DATA = {
    'scanqa': {
        'json_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/processed/scanqa_train_llava_style.json',
    },
    'sqa3d': {
        'json_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/processed/sqa3d_train_llava_style.json',
    },
    'scan2cap': {
        'json_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/processed/scan2cap_train_llava_style.json',
    },
    'scanrefer': {
        'json_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/processed/scanrefer_vg_train_llava_style.json',
    },
    'multi3drefer': {
        'json_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/processed/multi3drefer_train_llava_style.json',
    },
}

DEFAULT_VIDEO_FOLDER = '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/'
DEFAULT_ANNOTATION_DIR = '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/embodiedscan/'
DEFAULT_METADATA_DIR = '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/metadata/'


# ============================================================
# Mask 生成工具（与训练时 dataset_base_vae.py 中的 generate_mask_numpy 一致）
# ============================================================

def generate_mask_numpy(H, W, ratio, mode='rectangle', ar_range=(0.3, 3.0)):
    """
    在数据侧生成空间 mask（numpy 版本）。
    返回 (H, W) 的 numpy 数组，1=masked, 0=visible。
    与训练时 dataset_base_vae.py 中的实现完全一致。
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if ratio <= 0:
        return mask
    if ratio >= 1:
        return np.ones((H, W), dtype=np.uint8)

    total = H * W

    if mode == 'rectangle':
        target_area = max(1, int(round(ratio * total)))
        log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
        ar = math.exp(random.uniform(log_min, log_max))
        h = max(1, int(round(math.sqrt(target_area / ar))))
        w = max(1, int(round(ar * h)))
        h = max(1, min(h, H))
        w = max(1, min(w, W))
        top = 0 if H == h else random.randint(0, H - h)
        left = 0 if W == w else random.randint(0, W - w)
        mask[top:top+h, left:left+w] = 1

    elif mode == 'random':
        target = max(1, int(round(ratio * total)))
        noise = np.random.randn(H, W)
        flat = noise.flatten()
        threshold = np.partition(flat, -target)[-target]
        mask = (noise >= threshold).astype(np.uint8)

    elif mode == 'ellipse':
        target_area = max(1, int(round(ratio * total)))
        base_r = math.sqrt(target_area / math.pi)
        log_min, log_max = math.log(ar_range[0]), math.log(ar_range[1])
        ar = math.exp(random.uniform(log_min, log_max))
        ry = max(1, int(round(base_r / math.sqrt(ar))))
        rx = max(1, int(round(base_r * math.sqrt(ar))))
        ry, rx = min(ry, H // 2), min(rx, W // 2)
        cy = random.randint(ry, max(ry, H - ry - 1)) if H - 2 * ry > 0 else H // 2
        cx = random.randint(rx, max(rx, W - rx - 1)) if W - 2 * rx > 0 else W // 2
        yy, xx = np.mgrid[:H, :W]
        ellipse = ((yy - cy) ** 2 / (ry ** 2 + 1e-6) + (xx - cx) ** 2 / (rx ** 2 + 1e-6)) <= 1
        mask = ellipse.astype(np.uint8)

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return mask


# ============================================================
# 工具函数
# ============================================================

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def move_to_device(d, device):
    """将 dict 中所有 tensor 移动到指定 device。"""
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


def decode_latent(latent, h, w, vae_model, latent_patch_size=2, latent_channel=16):
    """将 flow-matching 输出的 latent 解码为 PIL Image。"""
    latent = latent.reshape(1, h, w, latent_patch_size, latent_patch_size, latent_channel)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, latent_channel, h * latent_patch_size, w * latent_patch_size)
    image = vae_model.decode(latent)
    image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
    image = Image.fromarray(image.to(torch.uint8).cpu().numpy())
    return image


def encode_image_to_latent(image, vae_model, vae_transform, latent_patch_size, latent_channel, latent_downsample, device):
    """
    将 PIL Image 通过 VAE encode 为 patchified latent。
    返回 (num_tokens, patch_dim) 的 tensor，以及 (h_latent, w_latent, H_img, W_img)。
    """
    image_tensor = vae_transform(image, img_num=1)
    H, W = image_tensor.shape[1:]
    h = H // latent_downsample
    w = W // latent_downsample

    # VAE encode
    image_tensor_batched = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        latent = vae_model.encode(image_tensor_batched)  # (1, C, H_latent, W_latent)

    # Patchify: (1, C, h*p, w*p) -> (h*w, p*p*C)
    p = latent_patch_size
    latent = latent[0, :, :h * p, :w * p]  # (C, h*p, w*p)
    latent = latent.reshape(latent_channel, h, p, w, p)
    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * latent_channel)

    return latent, h, w, H, W, image_tensor


def prepare_vae_ref_cache_input(
    gen_model,
    ref_image_tensors,
    ref_latent_shapes,
    vae_model,
    new_token_ids,
    device,
):
    """
    为 VAE ref 帧准备 forward_cache_update_vae 的输入参数。
    所有 ref 帧放在同一个 query 中（full attention）。

    Args:
        gen_model: Bagel 模型
        ref_image_tensors: list of (C, H, W) tensor, 经过 VAE transform 的 ref 帧
        ref_latent_shapes: list of (h, w), 每张 ref 帧的 latent shape
        vae_model: VAE 模型
        new_token_ids: 特殊 token id dict
        device: 设备

    Returns:
        cache_input: dict, forward_cache_update_vae 的输入参数
        total_kv_len: int, 注入 KV cache 后的总长度
    """
    packed_text_ids = []
    packed_text_indexes = []
    packed_vae_position_ids = []
    packed_vae_token_indexes = []
    packed_timesteps = []
    packed_position_ids = []
    packed_seqlens = []
    packed_indexes = []
    packed_key_value_indexes = []
    patchified_vae_latent_shapes = []

    query_curr = 0  # query 内的偏移
    curr = 0        # KV cache 内的偏移（初始为 0，因为 ref 是第一个注入的）

    for i, (img_tensor, (h, w)) in enumerate(zip(ref_image_tensors, ref_latent_shapes)):
        num_img_tokens = h * w
        patchified_vae_latent_shapes.append((h, w))

        # start_of_image
        packed_text_ids.append(new_token_ids['start_of_image'])
        packed_text_indexes.append(query_curr)
        packed_indexes.append(curr)
        curr += 1
        query_curr += 1

        # VAE position ids
        vae_pos_ids = gen_model.get_flattened_position_ids(
            img_tensor.size(1), img_tensor.size(2),
            gen_model.latent_downsample,
            max_num_patches_per_side=gen_model.max_latent_size,
        )
        packed_vae_position_ids.append(vae_pos_ids)

        # VAE token indexes
        packed_vae_token_indexes.extend(range(query_curr, query_curr + num_img_tokens))
        packed_indexes.extend(range(curr, curr + num_img_tokens))

        # timestep=0 表示 clean（训练时 ref 帧 timestep=-inf，sigmoid(-inf)≈0）
        packed_timesteps.extend([0.0] * num_img_tokens)

        curr += num_img_tokens
        query_curr += num_img_tokens

        # end_of_image
        packed_text_ids.append(new_token_ids['end_of_image'])
        packed_text_indexes.append(query_curr)
        packed_indexes.append(curr)
        curr += 1
        query_curr += 1

        # position_ids: 同一张图片内所有 token 共享同一个 rope position
        packed_position_ids.extend([i] * (num_img_tokens + 2))
        packed_seqlens.append(num_img_tokens + 2)

    # 构建 padded_images tensor
    image_sizes = [t.shape for t in ref_image_tensors]
    max_image_size = [max(s) for s in zip(*image_sizes)]
    padded_images = torch.zeros(len(ref_image_tensors), *max_image_size)
    for i, t in enumerate(ref_image_tensors):
        padded_images[i, :, :t.shape[1], :t.shape[2]] = t

    cache_input = {
        "padded_images": padded_images.to(device),
        "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
        "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0).to(device),
        "packed_timesteps": torch.tensor(packed_timesteps, dtype=torch.float32).to(device),
        "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long).to(device),
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long).to(device),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long).to(device),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long).to(device),
        "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int).to(device),
        "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long).to(device),
        "key_values_lens": torch.tensor([0], dtype=torch.int).to(device),  # 初始 KV cache 为空
        "packed_key_value_indexes": torch.tensor([], dtype=torch.long).to(device),
    }

    total_kv_len = curr
    num_ref = len(ref_image_tensors)
    new_rope = num_ref  # rope position 从 num_ref 开始

    return cache_input, total_kv_len, new_rope


@torch.no_grad()
def generate_nonref_images_prefix_vae(
    ref_images,
    nonref_images,
    question_text,
    answer_text,
    gen_model,
    vae_model,
    vae_transform,
    vit_transform,
    tokenizer,
    new_token_ids,
    device,
    # mask 参数（与训练一致）
    mask_modes=None,
    mask_ratios=None,
    # 生成参数
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=(0.0, 1.0),
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
    timestep_shift=3.0,
    num_timesteps=50,
):
    """
    VAE Prefix 模式（训推一致的 token-level masked reconstruction）。

    序列布局（与训练时 pack_sequence 一致）：
      [VAE ref (clean)] → [VAE nonref (denoise)] → [VIT all] → [text question] → [text answer]

    流程：
    1. 注入 VAE ref 帧的 clean latent 到 KV cache
    2. 注入 VIT 所有帧（ref + nonref）到 KV cache
    3. 注入文本 token（question + answer）到 KV cache
    4. 对所有 nonref 帧一起执行 flow-matching denoise

    Args:
        ref_images: list of PIL.Image, ref 视角图片
        nonref_images: list of PIL.Image, 待生成的 nonref 视角图片
        question_text: str, 原始 question 文本
        answer_text: str, 原始 answer 文本
        mask_modes: list of str, mask 模式列表
        mask_ratios: list of float, 对应的 mask 比例

    Returns:
        generated_images: list of PIL.Image, 生成的 nonref 图片
    """
    if mask_modes is None:
        mask_modes = ['random']
    if mask_ratios is None:
        mask_ratios = [0.75]

    latent_patch_size = gen_model.latent_patch_size
    latent_channel = gen_model.latent_channel
    latent_downsample = gen_model.latent_downsample

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # ============================================================
        # Step 0: 预计算所有帧的 clean latent 和 spatial mask
        # ============================================================
        # 整个 sample 共享一个 mask 参数（与训练一致）
        mask_idx = random.randint(0, len(mask_modes) - 1)
        cur_mask_mode = mask_modes[mask_idx]
        cur_mask_ratio = mask_ratios[mask_idx]

        # 编码 ref 帧
        ref_info = []
        for ref_img in ref_images:
            clean_latent, h_lat, w_lat, H_img, W_img, img_tensor = encode_image_to_latent(
                ref_img, vae_model, vae_transform,
                latent_patch_size, latent_channel, latent_downsample, device,
            )
            ref_info.append({
                'clean_latent': clean_latent,
                'h_lat': h_lat, 'w_lat': w_lat,
                'H_img': H_img, 'W_img': W_img,
                'image_tensor': img_tensor,
            })

        # 编码 nonref 帧并生成 mask
        nonref_info = []
        for nonref_img in nonref_images:
            clean_latent, h_lat, w_lat, H_img, W_img, img_tensor = encode_image_to_latent(
                nonref_img, vae_model, vae_transform,
                latent_patch_size, latent_channel, latent_downsample, device,
            )
            # 生成 VAE 空间 mask
            vae_mask = generate_mask_numpy(h_lat, w_lat, cur_mask_ratio, mode=cur_mask_mode)

            nonref_info.append({
                'clean_latent': clean_latent,
                'vae_mask': vae_mask,
                'h_lat': h_lat, 'w_lat': w_lat,
                'H_img': H_img, 'W_img': W_img,
                'image_tensor': img_tensor,
            })

        # ============================================================
        # Step 1: 注入 VAE ref 帧到 KV cache（clean latent, timestep=0）
        # ============================================================
        past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)

        ref_image_tensors = [info['image_tensor'] for info in ref_info]
        ref_latent_shapes = [(info['h_lat'], info['w_lat']) for info in ref_info]

        cache_input, total_kv_len, rope_pos = prepare_vae_ref_cache_input(
            gen_model, ref_image_tensors, ref_latent_shapes,
            vae_model, new_token_ids, device,
        )
        past_key_values = gen_model.forward_cache_update_vae(
            vae_model, past_key_values, **cache_input,
        )

        newlens = [total_kv_len]
        new_rope = [rope_pos]

        # ============================================================
        # Step 2: 注入 VIT 所有帧到 KV cache（ref + nonref，无 masking）
        # ============================================================
        all_images = ref_images + nonref_images
        for img in all_images:
            generation_input, newlens, new_rope = gen_model.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[img],
                transforms=vit_transform,
                new_token_ids=new_token_ids,
            )
            generation_input = move_to_device(generation_input, device)
            past_key_values = gen_model.forward_cache_update_vit(
                past_key_values, **generation_input,
            )

        # ============================================================
        # Step 3: 注入文本 token 到 KV cache
        # ============================================================
        # 3a. 注入 system prompt + question（去掉图片占位符）
        clean_question = question_text.replace('<image>', '').replace('<video>', '')
        clean_question = clean_question.replace('<vit_image>', '').replace('<vae_image>', '')
        clean_question = clean_question.replace('Reconstruct the scene', '').strip()

        text_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        if clean_question:
            text_prompt += clean_question

        generation_input, newlens, new_rope = gen_model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[text_prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_to_device(generation_input, device)
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

        # 3b. 注入 assistant prompt + answer
        assistant_prompt = '<|im_end|>\n<|im_start|>assistant\n'
        if answer_text:
            assistant_prompt += answer_text

        generation_input, newlens, new_rope = gen_model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[assistant_prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        generation_input = move_to_device(generation_input, device)
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

        # ============================================================
        # Step 4: VAE nonref 帧的 flow-matching denoise
        # ============================================================
        # 所有 nonref 帧一起 denoise（与训练时 video split full attention 一致）
        nonref_image_sizes = [(info['H_img'], info['W_img']) for info in nonref_info]

        # 准备 denoise query 输入
        generation_input = gen_model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=nonref_image_sizes,
            new_token_ids=new_token_ids,
        )
        generation_input = move_to_device(generation_input, device)

        # 准备 CFG 输入
        # cfg_text: 去掉文本 prompt 后的 condition（保留 VAE ref + VIT）
        cfg_text_past_key_values = copy.deepcopy(past_key_values)
        generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=nonref_image_sizes,
        )
        generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)

        # cfg_img: 完全空的 condition
        cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
        cfg_img_newlens = [0]
        cfg_img_new_rope = [0]
        generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            image_sizes=nonref_image_sizes,
        )
        generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)

        # 构建 mask_flat：所有 nonref 帧的 mask 拼接
        all_clean_latents = []
        all_mask_flats = []
        for info in nonref_info:
            mask_flat = torch.from_numpy(info['vae_mask'].flatten()).bool().to(device)
            all_mask_flats.append(mask_flat)
            all_clean_latents.append(info['clean_latent'])

        mask_flat_cat = torch.cat(all_mask_flats, dim=0)  # (total_nonref_tokens,)
        clean_latent_cat = torch.cat(all_clean_latents, dim=0)  # (total_nonref_tokens, patch_dim)

        # 初始化 x_t：被 mask 的 token 用纯噪声，未被 mask 的用 clean latent
        init_noise = generation_input['packed_init_noises']  # (total_nonref_tokens, patch_dim)
        x_t = clean_latent_cat.clone()
        init_noise = init_noise.to(dtype=x_t.dtype)
        x_t[mask_flat_cat] = init_noise[mask_flat_cat]

        # 提取 denoise 所需的参数
        packed_text_ids = generation_input['packed_text_ids']
        packed_text_indexes = generation_input['packed_text_indexes']
        packed_vae_position_ids = generation_input['packed_vae_position_ids']
        packed_vae_token_indexes = generation_input['packed_vae_token_indexes']
        packed_seqlens = generation_input['packed_seqlens']
        packed_position_ids = generation_input['packed_position_ids']
        packed_indexes = generation_input['packed_indexes']
        packed_key_value_indexes = generation_input['packed_key_value_indexes']
        key_values_lens = generation_input['key_values_lens']

        # Flow-matching denoise 循环
        timesteps = torch.linspace(1, 0, num_timesteps, device=device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for step_i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="denoise", leave=False):
            timestep = torch.tensor([t] * x_t.shape[0], device=device)

            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            v_t = gen_model._forward_flow(
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
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            )

            # 更新 x_t（velocity pointing from data to noise）
            x_t = x_t - v_t.to(x_t.device) * dts[step_i]

            # === 关键：每步后将未被 mask 的 token 重置为 clean latent ===
            x_t[~mask_flat_cat] = clean_latent_cat[~mask_flat_cat].to(dtype=x_t.dtype)

        # ============================================================
        # Step 5: VAE decode
        # ============================================================
        # 将 x_t 按每张 nonref 帧拆分
        generated_images = []
        offset = 0
        for info in nonref_info:
            h_lat = info['h_lat']
            w_lat = info['w_lat']
            num_tokens = h_lat * w_lat
            frame_latent = x_t[offset:offset + num_tokens]
            offset += num_tokens

            gen_img = decode_latent(
                frame_latent, h_lat, w_lat, vae_model,
                latent_patch_size=latent_patch_size,
                latent_channel=latent_channel,
            )
            generated_images.append(gen_img)

    return generated_images


def load_videollm3d_data(dataset_name, json_path=None):
    """
    加载 VideoLLM3D 数据集（scanqa / sqa3d / scan2cap / scanrefer / multi3drefer）。
    """
    if json_path is None:
        if dataset_name not in VIDEOLLM3D_DATA:
            raise ValueError(f"未知数据集: {dataset_name}，支持: {list(VIDEOLLM3D_DATA.keys())}")
        json_path = VIDEOLLM3D_DATA[dataset_name]['json_path']

    print(f"加载数据: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"共 {len(data)} 条数据")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="VAE Prefix multi-view generation on VideoLLM3D (ScanNet)."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存生成图片的目录")
    parser.add_argument("--dataset", type=str, default="scanqa",
                        choices=list(VIDEOLLM3D_DATA.keys()),
                        help="VideoLLM3D 数据集名称")
    parser.add_argument("--json_path", type=str, default=None,
                        help="自定义 JSON 数据文件路径（覆盖 --dataset 默认路径）")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="自定义模型权重文件路径（如训练后的 checkpoint），默认使用 model-path/ema.safetensors")
    # ── VideoProcessor 参数 ──
    parser.add_argument("--video_folder", type=str, default=DEFAULT_VIDEO_FOLDER,
                        help="视频数据根目录")
    parser.add_argument("--annotation_dir", type=str, default=DEFAULT_ANNOTATION_DIR,
                        help="embodiedscan 标注目录")
    parser.add_argument("--metadata_dir", type=str, default=DEFAULT_METADATA_DIR,
                        help="元数据目录")
    parser.add_argument("--frame_sampling_strategy", type=str, default="uniform",
                        help="帧采样策略（uniform / mc）")
    parser.add_argument("--force_sample", action="store_true",
                        help="是否强制采样指定帧数")
    parser.add_argument("--frames_upbound", type=int, default=32,
                        help="强制采样时的帧数上限")
    # ── ref/nonref 参数 ──
    parser.add_argument("--ref_num", type=int, default=-1,
                        help="每个场景的 ref 图片数量，-1 表示随机选择 N//4 或 N//2")
    parser.add_argument("--max_latent_size", type=int, default=64)
    # ── mask 参数（与训练脚本 train_sft_vae.sh 一致） ──
    parser.add_argument("--mask_mode", type=str, default="random,rectangle,ellipse",
                        help="mask 模式，逗号分隔（与训练时 --mask_mode 一致）")
    parser.add_argument("--mask_ratio", type=str, default="0.9,0.75,0.75",
                        help="mask 比例，逗号分隔（与训练时 --mask_ratio 一致）")
    # ── 生成参数 ──
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="评测的样本数量，-1 表示全部")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                        help="Text CFG scale")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                        help="Image CFG scale")
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                        choices=["global", "channel", "text_channel"],
                        help="CFG renorm 类型")
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    # ── VAE/VIT transform 参数（与训练时 joint_train.yaml 一致） ──
    parser.add_argument("--vae_max_image_size", type=int, default=256,
                        help="VAE transform max_image_size")
    parser.add_argument("--vae_min_image_size", type=int, default=256,
                        help="VAE transform min_image_size")
    parser.add_argument("--vit_max_image_size", type=int, default=384,
                        help="VIT transform max_image_size")
    parser.add_argument("--vit_min_image_size", type=int, default=256,
                        help="VIT transform min_image_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_mask_seed", type=int, default=None,
                        help="固定 mask 种子（为每个样本使用 seed + sample_idx），用于过拟合评测")
    args = parser.parse_args()

    # 解析 mask 参数
    mask_modes = args.mask_mode.split(',')
    mask_ratios = [float(r) for r in args.mask_ratio.split(',')]
    assert len(mask_modes) == len(mask_ratios), \
        f"mask_mode ({len(mask_modes)}) 和 mask_ratio ({len(mask_ratios)}) 数量不匹配"

    # ============================================================
    # 随机种子
    # ============================================================
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ============================================================
    # 分布式初始化
    # ============================================================
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"输出目录: {output_dir}")

    # ============================================================
    # 模型加载
    # ============================================================
    model_path = args.__dict__.get('model_path', args.__dict__.get('model-path'))
    if model_path is None:
        raise ValueError("必须指定 --model-path")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # VAE transform: 与训练时 joint_train.yaml 中 image_transform_args 一致
    vae_transform = ImageTransform(args.vae_max_image_size, args.vae_min_image_size, 16)
    # VIT transform: 与训练时 joint_train.yaml 中 vit_image_transform_args 一致
    vit_transform = ImageTransform(args.vit_max_image_size, args.vit_min_image_size, 14)

    if rank == 0:
        print(f"VAE transform: max={args.vae_max_image_size}, min={args.vae_min_image_size}, stride=16")
        print(f"VIT transform: max={args.vit_max_image_size}, min={args.vit_min_image_size}, stride=14")
        print(f"Mask modes: {mask_modes}, ratios: {mask_ratios}")

    # behind_vae=False: VAE prefix 模式（与训练时 BagelConfig 一致）
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
        behind_vae=False,  # VAE prefix 模式
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # 加载权重
    if args.model_weights is not None:
        weights_path = args.model_weights
    else:
        weights_path = os.path.join(model_path, "ema.safetensors")
    if rank == 0:
        print(f"加载模型权重: {weights_path}")
    model_state_dict = load_file(weights_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(f"模型加载信息: {msg}")
    del model_state_dict

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()

    # ============================================================
    # 初始化 VideoProcessor（与训练时一致）
    # ============================================================
    if rank == 0:
        print(f"初始化 VideoProcessor...")
        print(f"  video_folder: {args.video_folder}")
        print(f"  annotation_dir: {args.annotation_dir}")
        print(f"  metadata_dir: {args.metadata_dir}")
        print(f"  frame_sampling_strategy: {args.frame_sampling_strategy}")
        print(f"  force_sample: {args.force_sample}, frames_upbound: {args.frames_upbound}")

    video_processor = VideoProcessor(
        video_folder=args.video_folder,
        annotation_dir=args.annotation_dir,
        metadata_dir=args.metadata_dir,
        voxel_size=0.1,
        min_xyz_range=[-15, -15, -5],
        max_xyz_range=[15, 15, 5],
        frame_sampling_strategy=args.frame_sampling_strategy,
        val_box_type='pred',
    )

    # ============================================================
    # 加载数据
    # ============================================================
    data_items = load_videollm3d_data(args.dataset, args.json_path)

    if args.num_samples > 0:
        data_items = data_items[:args.num_samples]

    total_samples = len(data_items)
    samples_per_gpu = (total_samples + world_size - 1) // world_size
    start = rank * samples_per_gpu
    end = min(start + samples_per_gpu, total_samples)
    if rank == 0:
        print(f"总样本数: {total_samples}, 每个 GPU 处理约 {samples_per_gpu} 个样本")

    # ============================================================
    # 逐样本生成
    # ============================================================
    success_count = 0
    fail_count = 0

    for sample_idx in range(start, end):
        data_item = data_items[sample_idx]
        sample_id = data_item.get('id', f'sample_{sample_idx:06d}')
        video_id = data_item.get('video', '')
        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value'] if len(data_item['conversations']) > 1 else ''

        sample_dir = os.path.join(output_dir, f"{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        # 检查是否已完成（断点续评）
        done_flag = os.path.join(sample_dir, "done.txt")
        if os.path.exists(done_flag):
            print(f"[GPU {rank}] 跳过已完成的样本 {sample_idx}: {sample_id}")
            success_count += 1
            continue

        # ── 固定 mask 种子（过拟合评测） ──
        if args.fixed_mask_seed is not None:
            sample_seed = args.fixed_mask_seed + sample_idx
            random.seed(sample_seed)
            np.random.seed(sample_seed)

        # ── 使用 VideoProcessor 采样帧（与训练时一致） ──
        try:
            video_dict = video_processor.preprocess(
                video_id,
                force_sample=args.force_sample,
                frames_upbound=args.frames_upbound,
                generative_model_id=None,
                generative_feature_source='none',
            )
            raw_images = video_dict.pop("images")
        except Exception as e:
            print(f"[GPU {rank}] 样本 {sample_idx} ({video_id}) VideoProcessor 处理失败: {e}")
            fail_count += 1
            continue

        if len(raw_images) < 2:
            print(f"[GPU {rank}] 样本 {sample_idx} ({video_id}) 帧数不足: {len(raw_images)}")
            fail_count += 1
            continue

        # ── ref/nonref 分割（与训练时 videollm3d_dataset.py 一致） ──
        N = len(raw_images)
        if args.ref_num == -1:
            actual_ref_num = random.choice([max(1, N // 4), max(1, N // 2)])
            actual_ref_num = min(actual_ref_num, N - 1)
        else:
            actual_ref_num = min(args.ref_num, N - 1)

        ref_images = raw_images[:actual_ref_num]
        nonref_images = raw_images[actual_ref_num:]

        print(f"[GPU {rank}] 样本 {sample_idx} ({video_id}): "
              f"{actual_ref_num} ref + {len(nonref_images)} nonref 帧, "
              f"question: {question[:80]}...")

        # 保存 ref 图片
        for i, ref_img in enumerate(ref_images):
            ref_img.save(os.path.join(sample_dir, f"ref_{i:02d}.png"))

        # 保存 nonref GT 图片
        for i, nonref_img in enumerate(nonref_images):
            nonref_img.save(os.path.join(sample_dir, f"nonref_gt_{i:02d}.png"))

        # ── 生成 nonref 图片 ──
        try:
            generated = generate_nonref_images_prefix_vae(
                ref_images=ref_images,
                nonref_images=nonref_images,
                question_text=question,
                answer_text=answer,
                gen_model=model,
                vae_model=vae_model,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                device=device,
                mask_modes=mask_modes,
                mask_ratios=mask_ratios,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                cfg_interval=[0.0, 1.0],
                cfg_renorm_min=args.cfg_renorm_min,
                cfg_renorm_type=args.cfg_renorm_type,
                timestep_shift=args.timestep_shift,
                num_timesteps=args.num_timesteps,
            )
        except Exception as e:
            print(f"[GPU {rank}] 样本 {sample_idx} ({video_id}) 生成失败: {e}")
            traceback.print_exc()
            fail_count += 1
            continue

        # 保存生成的图片
        for i, gen_img in enumerate(generated):
            gen_img.save(os.path.join(sample_dir, f"nonref_gen_{i:02d}.png"))

        # 保存元信息
        meta = {
            "sample_idx": sample_idx,
            "sample_id": sample_id,
            "video_id": video_id,
            "dataset": args.dataset,
            "ref_num": actual_ref_num,
            "nonref_num": len(nonref_images),
            "total_frames": N,
            "question": question,
            "answer": answer,
            "mask_modes": mask_modes,
            "mask_ratios": mask_ratios,
            "mask_mode_used": mask_modes[0] if len(mask_modes) == 1 else "random_choice",
            "mode": "prefix_vae",
            "behind_vae": False,
        }
        with open(os.path.join(sample_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 标记完成
        with open(done_flag, "w") as f:
            f.write("done\n")

        success_count += 1
        print(f"[GPU {rank}] 样本 {sample_idx} ({video_id}) 完成，生成 {len(generated)} 张图片")

    print(f"[GPU {rank}] 所有任务完成: 成功 {success_count}, 失败 {fail_count}")
    dist.barrier()


if __name__ == "__main__":
    main()
