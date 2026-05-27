#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAE-style multi-view generation evaluation on RE10K dataset.

给定一个场景中的 ref 图片作为 condition（clean VAE + VIT），
通过 flow-matching denoising 生成 nonref 图片。

用法:
    torchrun --nproc_per_node=8 eval/gen/gen_images_mae_re10k.py \
        --model-path /path/to/model \
        --output_dir ./output/mae_re10k \
        --parquet_file /path/to/re10k_00000.parquet \
        --ref_num 2 \
        --max_frames 8 \
        --num_scenes 100
"""

import os
import copy
import json
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
import pyarrow.parquet as pq
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache


# ============================================================
# RE10K 数据集图片根目录（与训练时 edit_recon_mae_dataset.py 一致）
# ============================================================
RE10K_IMAGE_ROOT = (
    "/apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/"
    "datasets--mutou0308--RE10K/snapshots/"
    "85f1c43d30031e1cf9764eb30f40daa3ec72f6ae"
)


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


def sample_indices(num_imgs, max_frames, max_distance=10):
    """
    从 num_imgs 个视角中采样最多 max_frames 个索引。
    与训练时 edit_recon_mae_dataset.py 中的 _sample_indices 逻辑一致。
    """
    if num_imgs <= max_frames:
        return list(range(num_imgs))

    target_num = max_frames
    anchor = random.randint(0, num_imgs - 1)
    idxs = [anchor]

    scaled_max_distance = int(max_distance / 8 * target_num)
    start_idx = max(0, anchor - scaled_max_distance)
    end_idx = min(num_imgs - 1, start_idx + 2 * scaled_max_distance)
    start_idx = max(0, end_idx - 2 * scaled_max_distance)
    valid_indices = list(range(start_idx, end_idx + 1))

    if random.random() < 0.5:
        pool = [v for v in valid_indices if v != anchor]
        need = target_num - 1
        if len(pool) >= need:
            idxs.extend(random.sample(pool, need))
        else:
            idxs.extend(pool)
            remaining = [v for v in range(num_imgs) if v not in set(idxs)]
            still_need = target_num - len(idxs)
            idxs.extend(random.sample(remaining, min(still_need, len(remaining))))
    else:
        pool = sorted(valid_indices)
        num_additional = target_num - 1
        additional = []
        if len(pool) >= num_additional:
            indices_arr = list(pool)
            chunk_size = max(1, len(indices_arr) // (num_additional + 1))
            strata = []
            for i in range(0, len(indices_arr), chunk_size):
                strata.append(indices_arr[i:i + chunk_size])
            used = {anchor}
            for stratum in strata:
                candidates = [v for v in stratum if v not in used]
                if candidates and len(additional) < num_additional:
                    chosen = random.choice(candidates)
                    additional.append(chosen)
                    used.add(chosen)
            if len(additional) < num_additional:
                remaining = [v for v in pool if v not in used]
                still_need = num_additional - len(additional)
                additional.extend(random.sample(remaining, min(still_need, len(remaining))))
        else:
            additional = [v for v in pool if v != anchor]
            if len(additional) < num_additional:
                remaining = [v for v in range(num_imgs) if v not in set(idxs + additional)]
                still_need = num_additional - len(additional)
                additional.extend(random.sample(remaining, min(still_need, len(remaining))))
        idxs.extend(additional[:num_additional])

    return idxs


@torch.no_grad()
def generate_nonref_images(
    ref_images,
    nonref_images,
    gen_model,
    vae_model,
    vae_transform,
    vit_transform,
    tokenizer,
    new_token_ids,
    device,
    # 生成参数
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=(0.0, 1.0),
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
    timestep_shift=3.0,
    num_timesteps=50,
    max_image_size=1024,
    min_image_size=512,
):
    """
    给定 ref 图片作为 condition，逐张生成 nonref 图片。

    推理流程：
    1. 将所有 ref 图片的 clean VAE + VIT 注入 KV cache 作为 condition
    2. 对每张 nonref 图片，准备 VAE latent 并通过 flow-matching denoise 生成

    Args:
        ref_images: list of PIL.Image, ref 视角图片
        nonref_images: list of PIL.Image, 待生成的 nonref 视角图片（用于确定输出尺寸）
        其他参数: 模型和生成超参数

    Returns:
        generated_images: list of PIL.Image, 生成的 nonref 图片
    """
    generated_images = []

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # ============================================================
        # Step 1: 构建 condition context（ref 图片的 clean VAE + VIT）
        # ============================================================
        past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        for ref_img in ref_images:
            # 添加 clean VAE condition
            generation_input, newlens, new_rope = gen_model.prepare_vae_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[ref_img],
                transforms=vae_transform,
                new_token_ids=new_token_ids,
                timestep=0.0,  # clean VAE (t=0)
            )
            generation_input = move_to_device(generation_input, device)
            past_key_values = gen_model.forward_cache_update_vae(
                vae_model, past_key_values, **generation_input
            )

            # 添加 VIT condition
            generation_input, newlens, new_rope = gen_model.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[ref_img],
                transforms=vit_transform,
                new_token_ids=new_token_ids,
            )
            generation_input = move_to_device(generation_input, device)
            past_key_values = gen_model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        # ============================================================
        # Step 2: 逐张生成 nonref 图片
        # ============================================================
        for nonref_img in nonref_images:
            # 确定输出尺寸（与 nonref 原图一致，但限制在合理范围内）
            w_orig, h_orig = nonref_img.size
            scale = min(max_image_size / max(w_orig, h_orig), 1.0)
            scale = max(scale, min_image_size / min(w_orig, h_orig))
            w = round(w_orig * scale)
            h = round(h_orig * scale)
            # 确保能被 16 整除
            w = max(16, int(round(w / 16) * 16))
            h = max(16, int(round(h / 16) * 16))
            if max(w, h) > max_image_size:
                s = max_image_size / max(w, h)
                w = max(16, int(round(w * s / 16) * 16))
                h = max(16, int(round(h * s / 16) * 16))

            # --- cfg_text: 只有 ref 图片 condition，没有 text prompt ---
            # text CFG = 去掉 text prompt 后的 condition（这里没有 text，所以 cfg_text 就是当前 context）
            cfg_text_past_key_values = copy.deepcopy(past_key_values)
            generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                image_sizes=[(h, w)],
            )
            generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)

            # --- cfg_img: 没有任何 condition（空 context）---
            cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
            cfg_img_newlens = [0]
            cfg_img_new_rope = [0]
            generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_newlens,
                curr_rope=cfg_img_new_rope,
                image_sizes=[(h, w)],
            )
            generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)

            # --- 主 context: ref condition + 生成 nonref latent ---
            generation_input = gen_model.prepare_vae_latent(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                image_sizes=[(h, w)],
                new_token_ids=new_token_ids,
            )
            generation_input = move_to_device(generation_input, device)

            unpacked_latent = gen_model.generate_image(
                past_key_values=copy.deepcopy(past_key_values),
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            )

            # 解码 latent -> PIL Image
            latent = unpacked_latent[0]
            h_latent = h // gen_model.latent_downsample
            w_latent = w // gen_model.latent_downsample
            gen_img = decode_latent(
                latent, h_latent, w_latent, vae_model,
                latent_patch_size=gen_model.latent_patch_size,
                latent_channel=gen_model.latent_channel,
            )
            generated_images.append(gen_img)

    return generated_images


def load_scenes_from_parquet(parquet_file, image_root):
    """
    从 parquet 文件中读取所有场景数据。

    Returns:
        scenes: list of list of str, 每个场景包含一组图片路径
    """
    pf = pq.ParquetFile(parquet_file)
    scenes = []
    for rg_idx in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg_idx).to_pandas()
        for _, row in df.iterrows():
            image_paths = row["image_path"]
            if len(image_paths) < 2:
                continue
            # 转为绝对路径
            abs_paths = [os.path.join(image_root, p) for p in image_paths]
            scenes.append(abs_paths)
    return scenes


def main():
    parser = argparse.ArgumentParser(description="MAE-style multi-view generation on RE10K.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存生成图片的目录")
    parser.add_argument("--parquet_file", type=str,
                        default="/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/re10k_00000.parquet",
                        help="RE10K parquet 文件路径")
    parser.add_argument("--image_root", type=str, default=RE10K_IMAGE_ROOT,
                        help="RE10K 图片根目录")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="自定义模型权重文件路径（如训练后的 checkpoint），默认使用 model-path/ema.safetensors")
    parser.add_argument("--ref_num", type=int, default=2,
                        help="每个场景的 ref 图片数量")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="每个场景最多采样的视角数")
    parser.add_argument("--num_scenes", type=int, default=-1,
                        help="评测的场景数量，-1 表示全部")
    parser.add_argument("--max_latent_size", type=int, default=64)
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
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--min_image_size", type=int, default=512)
    parser.add_argument("--use_mae_masking", action="store_true", default=False,
                        help="是否在 BagelConfig 中启用 use_mae_masking（加载含 vit_mask_placeholder 的权重时需要）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    model_path = getattr(args, 'model_path', None) or args.__dict__.get('model-path')
    # argparse 中 --model-path 会被转为 model_path
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

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

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
        use_mae_masking=args.use_mae_masking,
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
    # 加载 RE10K 场景数据
    # ============================================================
    if rank == 0:
        print(f"从 {args.parquet_file} 加载场景数据...")
    scenes = load_scenes_from_parquet(args.parquet_file, args.image_root)
    if rank == 0:
        print(f"共加载 {len(scenes)} 个场景")

    if args.num_scenes > 0:
        scenes = scenes[:args.num_scenes]

    total_scenes = len(scenes)
    scenes_per_gpu = (total_scenes + world_size - 1) // world_size
    start = rank * scenes_per_gpu
    end = min(start + scenes_per_gpu, total_scenes)
    if rank == 0:
        print(f"每个 GPU 处理约 {scenes_per_gpu} 个场景")

    # ============================================================
    # 逐场景生成
    # ============================================================
    for scene_idx in range(start, end):
        scene_paths = scenes[scene_idx]
        scene_dir = os.path.join(output_dir, f"scene_{scene_idx:05d}")
        os.makedirs(scene_dir, exist_ok=True)

        # 检查是否已完成
        done_flag = os.path.join(scene_dir, "done.txt")
        if os.path.exists(done_flag):
            print(f"[GPU {rank}] 跳过已完成的场景 {scene_idx}")
            continue

        # 采样视角索引
        sampled_indices = sample_indices(len(scene_paths), args.max_frames)
        sampled_paths = [scene_paths[i] for i in sampled_indices]

        # 加载图片
        try:
            images = [pil_img2rgb(Image.open(p)) for p in sampled_paths]
        except Exception as e:
            print(f"[GPU {rank}] 场景 {scene_idx} 加载图片失败: {e}")
            continue

        # 随机打乱
        sampled_num = len(images)
        perm = list(range(sampled_num))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        sampled_paths = [sampled_paths[i] for i in perm]

        # 分割 ref / nonref
        ref_num = min(args.ref_num, sampled_num - 1)
        ref_images = images[:ref_num]
        nonref_images = images[ref_num:]

        print(f"[GPU {rank}] 场景 {scene_idx}: {ref_num} ref + {len(nonref_images)} nonref 图片")

        # 保存 ref 图片（作为参考）
        for i, ref_img in enumerate(ref_images):
            ref_img.save(os.path.join(scene_dir, f"ref_{i:02d}.png"))

        # 保存 nonref GT 图片（用于对比）
        for i, nonref_img in enumerate(nonref_images):
            nonref_img.save(os.path.join(scene_dir, f"nonref_gt_{i:02d}.png"))

        # 生成 nonref 图片
        try:
            generated = generate_nonref_images(
                ref_images=ref_images,
                nonref_images=nonref_images,
                gen_model=model,
                vae_model=vae_model,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                device=device,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                cfg_interval=[0.0, 1.0],
                cfg_renorm_min=args.cfg_renorm_min,
                cfg_renorm_type=args.cfg_renorm_type,
                timestep_shift=args.timestep_shift,
                num_timesteps=args.num_timesteps,
                max_image_size=args.max_image_size,
                min_image_size=args.min_image_size,
            )
        except Exception as e:
            print(f"[GPU {rank}] 场景 {scene_idx} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 保存生成的图片
        for i, gen_img in enumerate(generated):
            gen_img.save(os.path.join(scene_dir, f"nonref_gen_{i:02d}.png"))

        # 保存元信息
        meta = {
            "scene_idx": scene_idx,
            "ref_num": ref_num,
            "nonref_num": len(nonref_images),
            "sampled_indices": sampled_indices,
            "sampled_paths": sampled_paths,
            "perm": perm,
        }
        with open(os.path.join(scene_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # 标记完成
        with open(done_flag, "w") as f:
            f.write("done\n")

        print(f"[GPU {rank}] 场景 {scene_idx} 完成，生成 {len(generated)} 张图片")

    print(f"[GPU {rank}] 所有任务完成")
    dist.barrier()


if __name__ == "__main__":
    main()
