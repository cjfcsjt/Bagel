# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset_vae import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


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
            images.append(pil_img2rgb(Image.open(image_list[idx])))

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

        # 5. First pass: add ALL images (ref + nonref) as clean condition (no loss)
        #    - ref images: fully visible clean VAE + VIT condition
        #    - nonref images: clean VAE only (no VIT!) — masked positions in clean VAE
        #      will be replaced with learnable mask_placeholder in forward.
        #      VIT is NOT added for nonref images to prevent information leakage
        #      (VIT tokens are unmasked and would reveal the full image).
        for i, img in enumerate(images):
            if i < ref_num:
                # ref image: full clean VAE + VIT condition
                data = self._add_image(
                    data, img,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                    vae_type="ref",
                    enable_cfg=False,
                )
            else:
                # nonref image: clean VAE only (no VIT to avoid leaking masked regions)
                data = self._add_image(
                    data, img,
                    need_loss=False,
                    need_vae=True,
                    need_vit=False,
                    vae_type="nonref_clean",
                    enable_cfg=False,
                )

        # 6. Second pass: add nonref images as noise VAE tokens (with loss)
        #    need_vae=False, need_vit=False because clean condition copies
        #    were already added in the first pass above.
        #    These noise tokens attend to all preceding tokens (ref clean vae,
        #    ref vit, nonref masked clean vae) via attention for reconstruction.
        #    Note: nonref images have NO VIT in either pass to prevent leakage.
        for img in nonref_images:
            data = self._add_image(
                data, img,
                need_loss=True,   # mse loss on masked patches
                need_vae=False,   # clean vae condition already added in first pass
                need_vit=False,   # vit condition already added in first pass
                vae_type="nonref_noise",
                enable_cfg=False,
            )

        # 7. mask_ratio and mask_mode are now model attributes (BagelConfig),
        #    no longer passed from dataset.

        return data
