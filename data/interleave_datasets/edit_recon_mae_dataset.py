# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset_vae import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
import os

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte



class MAEMaskedReconIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """
    MAE-style masked reconstruction dataset for multi-view images.
    
    Data layout in the packed sequence (for one sample):
        --- Pass 1: ALL images' VIT tokens as condition (no loss) ---
        [ref_1_vit(ref_vit)] [ref_2_vit(ref_vit)] ... [nonref_1_vit(nonref_vit)] ...
        --- Pass 2: ALL images' VAE noise tokens in video mode (with loss) ---
        [ref_1_vae(ref_noise)] [ref_2_vae(ref_noise)] ... [nonref_1_vae(nonref_noise)] ...
        (all VAE frames share one timestep, bidirectional attention between frames)
    
    - VIT tokens (pass 1): ref_vit type → unmasked; nonref_vit type → masked in forward
      All VIT tokens are in "full" attention splits, visible to all subsequent tokens.
    - VAE noise tokens (pass 2): added via _add_video (video mode)
      → all frames in one "full" split with frame_delta, sharing one timestep
      → bidirectional attention between all VAE frames
      → can attend to all preceding VIT condition tokens
    """

    def __init__(self, ref_num=2, max_frames=8, **kwargs):
        super().__init__(**kwargs)
        self.ref_num = ref_num
        self.max_frames = max_frames

    def _sample_indices(self, num_imgs, max_distance=10):
        """
        Sample up to self.max_frames view indices from num_imgs available views.
        If num_imgs <= max_frames, use all views (no repeat/padding).

        Strategies (when num_imgs > max_frames):
            - Distance-based neighborhood sampling (50% chance)
            - Stratified sampling (50% chance)

        Args:
            num_imgs: total number of available views in the scene
            max_distance: max distance parameter for neighborhood sampling

        Returns:
            idxs: list of int, sampled view indices (length = min(num_imgs, max_frames))
        """
        if num_imgs <= self.max_frames:
            return list(range(num_imgs))

        target_num = self.max_frames

        # Pick a random anchor
        anchor = random.randint(0, num_imgs - 1)
        idxs = [anchor]

        scaled_max_distance = int(max_distance / 8 * target_num)
        start_idx = max(0, anchor - scaled_max_distance)
        end_idx = min(num_imgs - 1, start_idx + 2 * scaled_max_distance)
        start_idx = max(0, end_idx - 2 * scaled_max_distance)
        valid_indices = list(range(start_idx, end_idx + 1))

        if random.random() < 0.5:
            # Random neighborhood sampling (no replacement — never repeat)
            pool = [v for v in valid_indices if v != anchor]
            need = target_num - 1
            if len(pool) >= need:
                idxs.extend(random.sample(pool, need))
            else:
                # Pool too small: take all from pool, fill rest from remaining global indices
                idxs.extend(pool)
                remaining = [v for v in range(num_imgs) if v not in set(idxs)]
                still_need = target_num - len(idxs)
                idxs.extend(random.sample(remaining, min(still_need, len(remaining))))
        else:
            # Stratified sampling (no replacement)
            pool = sorted(valid_indices)
            num_additional = target_num - 1
            additional = []

            if len(pool) >= num_additional:
                strata = []
                # Split pool into roughly equal strata
                indices_arr = list(pool)
                chunk_size = max(1, len(indices_arr) // (num_additional + 1))
                for i in range(0, len(indices_arr), chunk_size):
                    strata.append(indices_arr[i:i + chunk_size])

                used = {anchor}
                for stratum in strata:
                    candidates = [v for v in stratum if v not in used]
                    if candidates and len(additional) < num_additional:
                        chosen = random.choice(candidates)
                        additional.append(chosen)
                        used.add(chosen)

                # If strata didn't yield enough, fill from remaining
                if len(additional) < num_additional:
                    remaining = [v for v in pool if v not in used]
                    still_need = num_additional - len(additional)
                    additional.extend(random.sample(remaining, min(still_need, len(remaining))))
            else:
                # Pool smaller than needed: take all, fill from global
                additional = [v for v in pool if v != anchor]
                if len(additional) < num_additional:
                    remaining = [v for v in range(num_imgs) if v not in set(idxs + additional)]
                    still_need = num_additional - len(additional)
                    additional.extend(random.sample(remaining, min(still_need, len(remaining))))

            idxs.extend(additional[:num_additional])

        assert len(idxs) == target_num, (
            f"Expected {target_num} frames, but got {len(idxs)}. num_imgs={num_imgs}"
        )
        return idxs

    def parse_row(self, row):
        image_list = row["image_path"]
        image_num = len(image_list)
        if image_num < 2:
            return None

        # 1. Sample indices: downsample if too many, keep all if fewer than max_frames
        sampled_indices = self._sample_indices(image_num)

        # 2. Load sampled images
        images = []
        for idx in sampled_indices:
            images.append(pil_img2rgb(Image.open(os.path.join("/apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/datasets--mutou0308--RE10K/snapshots/85f1c43d30031e1cf9764eb30f40daa3ec72f6ae", image_list[idx]))))

        # 3. Random shuffle
        sampled_num = len(images)
        perm = list(range(sampled_num))
        random.shuffle(perm)
        images = [images[i] for i in perm]

        # 4. Split into ref / non-ref (at least 1 non-ref)
        ref_num = min(self.ref_num, sampled_num - 1)
        ref_images = images[:ref_num]
        nonref_images = images[ref_num:]

        data = self._init_data()

        # 5. Pass 1: 按顺序添加所有图片的 VIT token 作为 condition（no loss）
        #    - ref 图片: vit_type="ref_vit"，forward 中不做 masking
        #    - nonref 图片: vit_type="nonref_vit"，forward 中对 masked 位置替换为 vit_mask_placeholder
        #    每个 VIT 是独立的 "full" split，可以被后续所有 token 看到。
        for i, img in enumerate(images):
            if i < ref_num:
                data = self._add_image(
                    data, img,
                    need_loss=False,
                    need_vae=False,
                    need_vit=True,
                    vit_type="ref_vit",
                    enable_cfg=False,
                )
            else:
                data = self._add_image(
                    data, img,
                    need_loss=False,
                    need_vae=False,
                    need_vit=True,
                    vit_type="nonref_vit",
                    enable_cfg=False,
                )

        # 6. Pass 2: 用 video 模式添加所有图片的 VAE noise token（with loss）
        #    所有帧合并为一个 "full" split，共享同一个 timestep，帧间双向注意力。
        #    这些 noised VAE token 可以看到前面所有的 VIT condition token。
        #    ref 图片: vae_type="ref_noise"
        #    nonref 图片: vae_type="nonref_noise"
        #    注意：_add_video 中所有帧共用同一个 vae_type，
        #    所以需要分别调用 ref 和 nonref，但仍然在同一个 video split 中。
        #    这里我们手动构建 video split 来支持不同的 vae_type。
        frame_indexes = list(range(len(images)))
        for i, (img, frame_idx) in enumerate(zip(images, frame_indexes)):
            vae_type = "ref_noise" if i < ref_num else "nonref_noise"
            current_sequence_plan = {
                'type': 'vae_image',
                'enable_cfg': 0,
                'loss': 1,
                'special_token_loss': 0,
                'special_token_label': None,
                'vae_type': vae_type,
                'split_start': i == 0,
                'split_end': i == len(images) - 1,
            }
            if i < len(frame_indexes) - 1:
                current_sequence_plan['frame_delta'] = frame_indexes[i + 1] - frame_idx
            data['sequence_plan'].append(current_sequence_plan)
            image_tensor = self.transform(img)
            height, width = image_tensor.shape[1:]
            data['image_tensor_list'].append(image_tensor)
            data['num_tokens'] += width * height // self.transform.stride ** 2

        # 7. mask_ratio and mask_mode are now model attributes (BagelConfig),
        #    no longer passed from dataset.

        return data
