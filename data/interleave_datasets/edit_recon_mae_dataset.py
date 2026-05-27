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



class MAEMaskedReconIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """
    Partial-noise multi-view consistency dataset.
    
    使用 apply_template_qwenvl2 模板构造文本 + VAE 图片序列。
    所有图片以 video 模式（共享 timestep、帧间双向注意力）添加到序列中。
    
    Data layout in the packed sequence (for one sample):
        [text tokens (template)] [ref_1_vae(ref)] [ref_2_vae(ref)] ... [nonref_1_vae(nonref)] ...
        (all VAE frames in one "full" split, sharing timestep, bidirectional attention)
    
    - ref 图片: vae_type='ref', loss=0（不参与去噪 loss，保持 clean）
    - nonref 图片: vae_type='nonref', loss=1（被 mask 的位置加噪去噪，计算 MSE loss）
    - 不使用 VIT token
    - mask 生成在 pack_sequence 中完成（数据打包层）
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

        # 5. 使用 apply_template_qwenvl2 构造模板化文本 + VAE 图片序列
        question = 'Reconstruct the scene' + '<vae_image>' * len(images)
        answer = ''
        split_list = apply_template_qwenvl2(
            question_with_image_tokens=question, answer=answer
        )

        data = self._init_data()

        # 6. 遍历 split_list，构造 sequence_plan
        #    - text 类型：tokenize 并加入 text_ids_list + sequence_plan
        #    - vae 类型：以 video 模式添加 VAE 图片（ref/nonref 区分）
        vae_counter = 0  # 追踪当前是第几张 VAE 图片
        total_vae_count = len(images)
        frame_indexes = list(range(total_vae_count))

        for item in split_list:
            if item['type'] == 'text':
                text_data = item['value']
                text_ids = self.tokenizer.encode(text_data)
                if len(text_ids) > 0:
                    data['text_ids_list'].append(text_ids)
                    data['num_tokens'] += len(text_ids)
                    current_plan = {
                        'type': 'text',
                        'enable_cfg': 0,
                        'loss': int(item['loss']),
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                    data['sequence_plan'].append(current_plan)

            elif item['type'] == 'vae':
                # 当前图片
                img = images[vae_counter]
                frame_idx = frame_indexes[vae_counter]

                # 区分 ref / nonref
                if vae_counter < ref_num:
                    vae_type = 'ref'
                    loss = 0
                else:
                    vae_type = 'nonref'
                    loss = 1

                # video 模式：split_start / split_end / frame_delta
                current_sequence_plan = {
                    'type': 'vae_image',
                    'enable_cfg': 0,
                    'loss': loss,
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'vae_type': vae_type,
                    'split_start': vae_counter == 0,
                    'split_end': vae_counter == total_vae_count - 1,
                }
                if vae_counter < total_vae_count - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[vae_counter + 1] - frame_idx

                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(img)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

                vae_counter += 1

        return data
