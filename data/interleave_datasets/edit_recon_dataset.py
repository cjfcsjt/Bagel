# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset_vae import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb, apply_template_qwenvl2
import os

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte



class MaskedReconIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """
    Masked reconstruction dataset for multi-view images.
    
    Data layout in the packed sequence (for one sample):
        --- Pass 1: ALL images as clean condition (no loss) ---
        [ref_1_vae_clean][ref_1_vit] ... [ref_K_vae_clean][ref_K_vit]
        [nonref_1_vae_clean] ... [nonref_M_vae_clean]
        --- Pass 2: nonref images as noise VAE (with loss) ---
        [nonref_1_vae_noise] ... [nonref_M_vae_noise]
    
    - ref images (pass 1):    need_vae=True, need_vit=True, need_loss=False
      → fully visible clean condition tokens (VAE + VIT)
    - nonref images (pass 1): need_vae=True, need_vit=False, need_loss=False
      → clean VAE only; masked positions get learnable mask_placeholder in forward
      → NO VIT to prevent information leakage of masked regions
    - nonref images (pass 2): need_loss=True, need_vae=False, need_vit=False
      → noise split (fully noised via flow-matching in forward);
        attends to ALL preceding clean condition tokens for reconstruction
    """

    def __init__(self, ref_num=2, max_frames=8, behind_vae=False, **kwargs):
        super().__init__(**kwargs)
        self.ref_num = ref_num
        self.max_frames = max_frames
        self.behind_vae = behind_vae

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

        # 5. 根据模式分支
        if self.behind_vae:
            return self._parse_row_behind_vae(images, ref_num)
        else:
            return self._parse_row_two_pass(images, ref_num)

    def _parse_row_two_pass(self, images, ref_num):
        """
        原始的两遍模式（two-pass）数据构造。

        Data layout:
            --- Pass 1: ALL images as clean condition (no loss) ---
            [ref_1_vae_clean][ref_1_vit] ... [ref_K_vae_clean][ref_K_vit]
            [nonref_1_vae_clean] ... [nonref_M_vae_clean]
            --- Pass 2: nonref images as noise VAE (with loss) ---
            [nonref_1_vae_noise] ... [nonref_M_vae_noise]
        """
        nonref_images = images[ref_num:]
        data = self._init_data()

        # Pass 1: add ALL images (ref + nonref) as clean condition (no loss)
        for i, img in enumerate(images):
            if i < ref_num:
                data = self._add_image(
                    data, img,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                    vae_type="ref",
                    enable_cfg=False,
                )
            else:
                data = self._add_image(
                    data, img,
                    need_loss=False,
                    need_vae=True,
                    need_vit=False,
                    vae_type="nonref_clean",
                    enable_cfg=False,
                )

        # Pass 2: add nonref images as noise VAE tokens (with loss)
        for img in nonref_images:
            data = self._add_image(
                data, img,
                need_loss=True,
                need_vae=False,
                need_vit=False,
                vae_type="nonref_noise",
                enable_cfg=False,
            )

        return data

    def _parse_row_behind_vae(self, images, ref_num):
        """
        behind_vae 模式的数据构造。

        Data layout:
            [text: system + user prompt]
              [VIT: ref_1] [VIT: ref_2] ... [VIT: ref_K]
              [VIT: nonref_1] [VIT: nonref_2] ... [VIT: nonref_M]
              [VAE: nonref_1 (masked denoise)] ... [VAE: nonref_M (masked denoise)]
            [text: assistant]

        - 所有帧（ref + nonref）都有 VIT token
        - 只有 nonref 帧有 VAE token（放在 VIT 之后），做 masked denoise
        - ref VIT: vit_type='ref'，完全可见
        - nonref VIT: vit_type='nonref'，被 mask 的 VIT token 对 VAE 不可见
        - nonref VAE: vae_type='nonref'，loss=1，video 模式
        """
        total_count = len(images)
        nonref_count = total_count - ref_num

        # 构造 question: VIT tokens for all + VAE tokens for nonref only
        question = 'Reconstruct the scene' + '<vit_image>' * total_count + '<vae_image>' * nonref_count
        answer = ''
        split_list = apply_template_qwenvl2(
            question_with_image_tokens=question, answer=answer
        )

        data = self._init_data()

        vit_counter = 0   # 追踪当前处理到第几个 VIT image
        vae_counter = 0   # 追踪当前处理到第几个 VAE image
        frame_indexes = list(range(nonref_count))

        for item in split_list:
            if item['type'] == 'text':
                text_data = item['value']
                text_ids = self.tokenizer.encode(text_data)
                if len(text_ids) > 0:
                    data['text_ids_list'].append(text_ids)
                    data['num_tokens'] += len(text_ids)
                    data['sequence_plan'].append({
                        'type': 'text',
                        'enable_cfg': 0,
                        'loss': int(item['loss']),
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })

            elif item['type'] == 'vit':
                # VIT token: ref 或 nonref
                img = images[vit_counter]
                vit_type = 'ref' if vit_counter < ref_num else 'nonref'

                data['sequence_plan'].append({
                    'type': 'vit_image',
                    'enable_cfg': 0,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'vit_type': vit_type,
                })
                vit_image_tensor = self.vit_transform(img)
                height, width = vit_image_tensor.shape[1:]
                data['image_tensor_list'].append(vit_image_tensor)
                data['num_tokens'] += width * height // self.vit_transform.stride ** 2

                vit_counter += 1

            elif item['type'] == 'vae':
                # VAE token: 只有 nonref 帧，video 模式
                nonref_idx = vae_counter
                img = images[ref_num + nonref_idx]  # nonref 图片从 images[ref_num] 开始

                current_sequence_plan = {
                    'type': 'vae_image',
                    'enable_cfg': 0,
                    'loss': 1,
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'vae_type': 'nonref',
                    'split_start': vae_counter == 0,
                    'split_end': vae_counter == nonref_count - 1,
                }
                if vae_counter < nonref_count - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[vae_counter + 1] - frame_indexes[vae_counter]

                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(img)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

                vae_counter += 1

        return data
