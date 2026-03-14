"""
Reconstruction Dataset in Parquet format.

This module refactors the original JSONL-based SftJSONLIterableReconDataset (recon_dataset.py)
into the ParquetStandardIterableDataset pattern used by recon_then_und_dataset.py.

Key changes from recon_dataset.py:
- Inherits ParquetStandardIterableDataset (Parquet-based data loading)
- Uses parse_row() instead of __iter__() for data processing
- Uses modular _init_data / _add_text / _add_image methods
- Preserves all BlendedMVS sampling strategies, depth loading, 3D point cloud generation
"""

import io
import re
import json
import random
import os
import os.path as osp
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb, apply_template_qwenvl2_reconThenUnd
from ..distributed_iterable_dataset import DistributedIterableDataset
from ..dataset_utils_vggt import *
import torch
import torch.distributed as dist
from copy import deepcopy
import cv2
import modeling.pi3.utils.cropping as cropping
from modeling.pi3.utils.geometry import depthmap_to_absolute_camera_coordinates

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class ReconParquetIterableDataset(ParquetStandardIterableDataset, DistributedIterableDataset):
    """
    Parquet-based reconstruction dataset.

    This dataset loads scene data from Parquet files and processes them for 3D reconstruction.
    It follows the same pattern as ReconthenUndIterableDataset but is dedicated to recon-only tasks.

    Data format (per row in Parquet):
        - scene_name: str, dataset name (e.g., 'blendmvs', 'scannet')
        - seq_name: str, sequence/scene identifier
        - img_dir: str, path to the scene directory
        - num_images: int, number of available images
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_size = 518  # DINOv2 default
        self.patch_size = 14
        self.aug_scale = [0.8, 1.2]
        self.rescale = True
        self.rescale_aug = True
        self.landscape_check = False
        self.training = True
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.random_sample_thres = 0.1
        self.aug_crop = 16
        self.aug_focal = 0.9
        self.z_far = 0
        self.frame_num = 8
        self.random_image_num = 0
        self.random_aspect_ratio = 1.0
        self.resolution = [518, 518]
        self.shuffle_seq_views = True

    # ────────────────────────── Setters ──────────────────────────

    def set_random_image_num(self, num):
        self.random_image_num = num
        self.frame_num = num

    def set_random_aspect_ratio(self, num):
        self.random_aspect_ratio = num

    def set_step_rng(self, rng):
        self._rng = np.random.default_rng(rng)

    # ────────────────────────── Utility Methods ──────────────────────────

    def pop_first(self, arr):
        first_val = arr[0]
        new_arr = arr[1:]
        return first_val, new_arr

    def blender2opencv_c2w(self, pose):
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()

    def _load_pfm_file(self, file_path):
        """
        Load a PFM (Portable Float Map) depth map file.
        Used for BlendedMVS dataset depth maps.
        """
        with open(file_path, "rb") as file:
            header = file.readline().decode("UTF-8").strip()

            if header == "PF":
                is_color = True
            elif header == "Pf":
                is_color = False
            else:
                raise ValueError(f"Not a valid PFM file: {file_path}")

            dimensions = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
            if dimensions:
                img_width, img_height = map(int, dimensions.groups())
            else:
                raise ValueError(f"Invalid PFM header format: {file_path}")

            endian_scale = float(file.readline().decode("UTF-8").strip())
            if endian_scale < 0:
                dtype = "<f"  # little-endian
            else:
                dtype = ">f"  # big-endian

            data_buffer = file.read()
            img_data = np.frombuffer(data_buffer, dtype=dtype)

            if is_color:
                img_data = np.reshape(img_data, (img_height, img_width, 3))
            else:
                img_data = np.reshape(img_data, (img_height, img_width))

            img_data = cv2.flip(img_data, 0)

        return img_data

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """
        Downsizes the image with LANCZOS interpolation, crops centered on the principal point,
        and applies optional focal augmentation.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f'Bad principal point in view={info}'
        assert min_margin_y > H / 5, f'Bad principal point in view={info}'

        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox, normal=normal
        )

        W, H = image.size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5)
            image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(
                image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask
            )

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask
        )

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other

    def get_target_shape(self, aspect_ratio):
        """Calculate the target shape based on the given aspect ratio."""
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    # ────────────────────────── Data Structure Methods ──────────────────────────

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'image_grid_thw_list': [],
            'dino_images': [],
            'depths': [],
            'extrinsics': [],
            'intrinsics': [],
            'cam_points': [],
            'world_points': [],
            'point_masks': [],
            'original_sizes': [],
            'dino_image_tensor_list': [],
            'dino_thw': [],
            'img_per_seq': 0,
            'num_tokens': 0,
            'new_depths': [],
            'view_infos': [],
            'image_paths': [],
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': 0,
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, dino_meta, need_loss, need_dino, need_vit,
                   enable_cfg=True, rng=None, view_info=None,
                   has_3d_annotation=True, split_start=True, split_end=True):
        """
        Add an image to the data dictionary.

        For recon-only, we always use need_dino=True, need_vit=False.
        The image, depthmap, extrinsics, intrinsics are already loaded and passed via dino_meta.
        """
        assert need_loss or need_dino or need_vit

        if need_dino:
            if has_3d_annotation:
                # For recon dataset, 3D data is already loaded and preprocessed in parse_row
                # dino_meta contains pre-loaded: depth_map, extri_opencv, intrinsic_, image (PIL)
                depth_map = dino_meta['depth_map']
                extri_opencv = dino_meta['extri_opencv']
                intrinsic_ = dino_meta['intrinsic_']
                original_size = dino_meta.get('original_size', np.array(image.size[::-1]) if isinstance(image, PIL.Image.Image) else np.array(image.shape[:2]))

                data['dino_images'].append(image)
                data['depths'].append(depth_map.astype(np.float32))
                data['extrinsics'].append(extri_opencv.astype(np.float32))
                data['intrinsics'].append(intrinsic_.astype(np.float32))
                data['original_sizes'].append(original_size)
                data['view_infos'].append(view_info)

                # Determine z_far based on scene type
                scene_label = dino_meta.get('scene_name', '')
                if scene_label in ['co3dv2', 'wildrgbd', 'blendmvs', 'blendedmvs']:
                    z_far = 0
                elif scene_label in ['gtasfm', 'matrixcity', 'taskonomy', 'hypersim', 'nav_20w', 'vkitti', 'megadepth', 'dl3dv', 'omniworld', 'unreal4k']:
                    z_far = 0
                elif scene_label in ['tartanair', 'scannet']:
                    z_far = 80
                elif scene_label in ['scannetpp', 'arkitscenes']:
                    z_far = 120
                else:
                    z_far = 0

                assert np.isfinite(extri_opencv).all(), f'NaN in camera pose for view {view_info}'
                assert np.isfinite(depth_map).all(), f'NaN in depthmap for view {view_info}'

                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                    depth_map, intrinsic_, extri_opencv, z_far=z_far
                )
                valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
                depth_map[~valid_mask] = 0.0
                assert valid_mask.sum() > 0, f"viewinfo{view_info}, depthmap{depth_map}"

                data['new_depths'].append(depth_map)
                data['world_points'].append(pts3d)
                data['point_masks'].append(valid_mask)
            else:
                # No 3D annotation: placeholder data
                if not isinstance(image, PIL.Image.Image):
                    image = PIL.Image.fromarray(image)
                # image = image.resize((self.resolution[1], self.resolution[0]), PIL.Image.LANCZOS)
                data['dino_images'].append(image)
                data['view_infos'].append(view_info)
                data['original_sizes'].append(np.array(self.resolution))

                data['depths'].append(np.zeros((self.resolution[0], self.resolution[1]), dtype=np.float32))
                data['extrinsics'].append(np.eye(4, dtype=np.float32))
                data['intrinsics'].append(np.eye(3, dtype=np.float32))
                data['new_depths'].append(np.zeros((self.resolution[0], self.resolution[1]), dtype=np.float32))
                data['world_points'].append(np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.float32))
                data['point_masks'].append(np.zeros((self.resolution[0], self.resolution[1]), dtype=bool))

            # DINO transform
            transform_stride = 14  # hardcode
            image_tensor = self.dino_transform(image, img_num=1)
            data['dino_image_tensor_list'].append(image_tensor)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // transform_stride ** 2

            grid_t = 1
            grid_h, grid_w = height // self.patch_size, width // self.patch_size
            thw_dino = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)
            data['dino_thw'].append(thw_dino)

            data['sequence_plan'].append(
                {
                    'type': 'dino_image',
                    'enable_cfg': 0,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': split_start,
                    'split_end': split_end,
                }
            )

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': 0,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor, image_grid_thw = self.vit_transform([image], img_num=1)
            data['num_tokens'] += vit_image_tensor.shape[0] // 4
            data['image_tensor_list'].append(vit_image_tensor)
            data['image_grid_thw_list'].append(image_grid_thw[0])

        return data

    # ────────────────────────── View Sampling (Shared) ──────────────────────────

    def _sample_view_indices(self, num_imgs, rng, max_distance=10):
        """
        Sample self.frame_num view indices from num_imgs available views.

        Strategies:
            - Fully random sampling (when frame_num > 16 and random threshold met)
            - Distance-based neighborhood sampling (50% chance)
            - Stratified sampling (50% chance)

        Args:
            num_imgs: total number of available views in the scene
            rng: numpy random generator
            max_distance: max distance parameter for distance-based sampling
                          (scene-specific, e.g. blendmvs=10, scannet=20)

        Returns:
            idxs: list of int, sampled view indices of length self.frame_num
        """
        if self.frame_num > 16 and rng.random() < self.random_sample_thres:
            # Fully random sampling
            should_replace = num_imgs < self.frame_num
            idxs = list(rng.choice(range(num_imgs), size=self.frame_num, replace=should_replace))
        else:
            # Distance-based sampling: pick a random anchor
            idxs = [rng.integers(0, num_imgs)]
            scaled_max_distance = int(max_distance / 8 * self.frame_num)

            start_idx = max(0, idxs[-1] - scaled_max_distance)
            end_idx = min(num_imgs - 1, start_idx + 2 * scaled_max_distance)
            start_idx = max(0, end_idx - 2 * scaled_max_distance)
            valid_indices = np.arange(start_idx, end_idx + 1)

            if rng.random() < 0.5:
                # Random neighborhood sampling
                should_replace = len(valid_indices) < self.frame_num - 1
                idxs.extend(list(rng.choice(valid_indices, size=self.frame_num - 1, replace=should_replace)))
            else:
                # Stratified sampling
                ref_frame_val = idxs[0]
                num_additional_to_select = self.frame_num - 1
                additional_selected_values = []
                pool_for_others_values = sorted(list(valid_indices))

                should_replace_for_others = len(pool_for_others_values) < num_additional_to_select

                if not pool_for_others_values:
                    if should_replace_for_others:
                        additional_selected_values = [ref_frame_val] * num_additional_to_select
                else:
                    if not should_replace_for_others and len(pool_for_others_values) >= num_additional_to_select:
                        strata = np.array_split(pool_for_others_values, num_additional_to_select + 1)
                        for stratum in strata:
                            if len(stratum) > 0 and ref_frame_val not in stratum:
                                additional_selected_values.append(rng.choice(stratum))
                    else:
                        additional_selected_values = list(rng.choice(
                            pool_for_others_values,
                            num_additional_to_select,
                            replace=(should_replace_for_others or (len(pool_for_others_values) < num_additional_to_select))
                        ))

                idxs = [ref_frame_val, *additional_selected_values]

        assert len(idxs) == self.frame_num, (
            f"Expected {self.frame_num} frames, but got {len(idxs)} frames. idxs: {idxs}"
        )
        return idxs

    # ────────────────────────── Postprocessing (Shared) ──────────────────────────

    def _postprocess_views(self, raw_views, rng):
        """
        Apply crop/resize preprocessing to raw views that have 3D annotations.
        Shared across all scene types.

        Args:
            raw_views: list of dicts, each with keys:
                - image: np.ndarray (H, W, 3) RGB
                - depth: np.ndarray (H, W) depth map
                - extrinsic: np.ndarray (4, 4) cam2world
                - intrinsic: np.ndarray (3, 3)
                - view_info: str
            rng: numpy random generator

        Returns:
            images: list of PIL images (after crop/resize)
            depths: list of depth maps (after crop/resize)
            extrinsics: list of 4x4 extrinsic matrices (unchanged)
            intrinsics: list of 3x3 intrinsic matrices (after crop/resize)
            view_infos: list of view info strings
        """
        images, depths, extrinsics, intrinsics, view_infos = [], [], [], [], []

        for v in raw_views:
            img, dep, intri = self._crop_resize_if_necessary(
                v['image'], v['depth'], v['intrinsic'].copy(),
                self.resolution, rng=rng, info=v['view_info']
            )
            images.append(img)
            depths.append(dep)
            extrinsics.append(v['extrinsic'].astype(np.float32))
            intrinsics.append(intri.astype(np.float32))
            view_infos.append(v['view_info'])

        return images, depths, extrinsics, intrinsics, view_infos

    def _simple_resize_views(self, image_paths, idxs, scene_label, scene_id):
        """
        Simple image loading and resize for views without 3D annotations.
        No depth map, no intrinsics/extrinsics — just load RGB and resize.

        Args:
            image_paths: list of all available image paths
            idxs: list of sampled indices
            scene_label: str, scene name for view_info
            scene_id: str, sequence name for view_info

        Returns:
            images: list of PIL images (resized to self.resolution)
            depths: list of zero arrays (placeholder)
            extrinsics: list of identity matrices (placeholder)
            intrinsics: list of identity matrices (placeholder)
            view_infos: list of view info strings
        """
        images, depths, extrinsics, intrinsics, view_infos = [], [], [], [], []
        h, w = self.resolution

        for idx in idxs:
            img_path = image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            image = image.resize((w, h), Image.LANCZOS)

            images.append(image)
            depths.append(np.zeros((h, w), dtype=np.float32))
            extrinsics.append(np.eye(4, dtype=np.float32))
            intrinsics.append(np.eye(3, dtype=np.float32))
            view_infos.append(f"{scene_label}/{scene_id}/{idx}")

        return images, depths, extrinsics, intrinsics, view_infos

    # ────────────────────────── Scene I/O Methods (Per-scene) ──────────────────────────

    def _get_blendmvs_scene_info(self, data_item):
        """
        Get scene metadata for BlendedMVS: view names, total count, image paths.
        Pure metadata — no data loading.

        Returns:
            scene_info: dict with 'all_view_names', 'num_imgs', 'scene_dir', 'cam_dir'
        """
        scene_dir = data_item.get('img_dir', None)
        scene_dir = scene_dir.replace('processed/', '')
        if scene_dir is None:
            raise ValueError(f"BlendedMVS scene_dir not found in data_item")

        cam_dir = os.path.join(scene_dir, 'cams')
        all_view_names = sorted([
            f[:-8] for f in os.listdir(cam_dir)
            if not f.startswith('pair') and f.endswith('_cam.txt')
        ])

        return {
            'all_view_names': all_view_names,
            'num_imgs': len(all_view_names),
            'scene_dir': scene_dir,
            'cam_dir': cam_dir,
        }

    def _load_blendmvs_raw(self, data_item, scene_info, idxs):
        """
        Load raw BlendedMVS views by indices. Only does file I/O, no sampling or crop/resize.

        Args:
            data_item: dict with scene metadata
            scene_info: dict from _get_blendmvs_scene_info
            idxs: list of view indices to load

        Returns:
            raw_views: list of dicts with keys: image, depth, extrinsic, intrinsic, view_info
        """
        data_scene_name = data_item['scene_name']
        this_scene = data_item['seq_name']
        scene_dir = scene_info['scene_dir']
        cam_dir = scene_info['cam_dir']
        all_view_names = scene_info['all_view_names']

        raw_views = []
        for idx in idxs:
            view_name = all_view_names[idx]

            # Load camera parameters
            cam_file = os.path.join(cam_dir, f'{view_name}_cam.txt')
            with open(cam_file, 'r') as f:
                f.readline()  # skip extrinsic tag
                RT_world2cam = np.array(
                    [list(map(float, f.readline().split())) for _ in range(4)],
                    dtype=np.float32
                )
                extri_opencv = np.linalg.inv(RT_world2cam)  # cam2world

                f.readline()  # skip blank line
                f.readline()  # skip intrinsic tag
                intri_opencv = np.array(
                    [list(map(float, f.readline().split())) for _ in range(3)],
                    dtype=np.float32
                )

            # Load RGB image
            image_path = os.path.join(scene_dir, "blended_images", f"{view_name}.jpg")
            rgb_image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # Load depth map (PFM format)
            depth_path = os.path.join(scene_dir, "rendered_depth_maps", f"{view_name}.pfm")
            depth_map = self._load_pfm_file(depth_path)

            raw_views.append({
                'image': rgb_image,
                'depth': depth_map,
                'extrinsic': extri_opencv,
                'intrinsic': intri_opencv,
                'view_info': f"{data_scene_name}/{this_scene}/{view_name}",
            })

        return raw_views

    # ────────────────────────── Main Parse Method ──────────────────────────

    def parse_row(self, row):
        """
        Parse a single row from the Parquet dataset.

        Expected row format (same as the JSONL format from recon_dataset.py):
            - scene_name: str (e.g., 'blendmvs', 'scannet')
            - seq_name: str
            - img_dir: str (path to scene directory)
            - num_images: int

        Returns:
            data: dict with all processed data for this sample
        """
        rng = self._rng
        data_scene_name = row['scene_name']
        # this_scene = row['seq_name']
        this_scene = row['image_path'][0].split('/')[-2]
        has_3d_annotation = (
            'depth_list' in row and row['depth_list'] is not None and len(row.get('depth_list', [])) > 0 and
            'poses' in row and row['poses'] is not None and len(row.get('poses', [])) > 0
        )
        
        # Convert row to dict for scene loading methods
        data_item = {
            'scene_name': data_scene_name,
            # 'seq_name': this_scene,
            # 'img_dir': row.get('img_dir', ''),
            'num_images':  len(row['image_path']),
        }

        # ── 1. Load views based on scene type ──
        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        view_infos = []
        image_paths = []

        if has_3d_annotation:
            # Step 1a: Get scene metadata (view names, total count, paths)
            if data_scene_name == 'blendmvs':
                scene_info = self._get_blendmvs_scene_info(data_item)
                max_distance = 10  # scene-specific max_distance for sampling
            elif data_scene_name == 'scannet':
                # TODO: Add scannet scene info loading
                # scene_info = self._get_scannet_scene_info(data_item)
                # max_distance = 20
                return []
            else:
                raise ValueError(f"Unsupported scene type: {data_scene_name}")

            # Step 1b: Sample view indices (shared across all scenes)
            num_scene_imgs = scene_info['num_imgs']
            if num_scene_imgs != self.frame_num:
                idxs = self._sample_view_indices(num_scene_imgs, rng, max_distance=max_distance)
            else:
                idxs = list(range(num_scene_imgs))

            # Step 1c: Load raw data by indices (scene-specific I/O only)
            if data_scene_name == 'blendmvs':
                raw_views = self._load_blendmvs_raw(data_item, scene_info, idxs)
            elif data_scene_name == 'scannet':
                # raw_views = self._load_scannet_raw(data_item, scene_info, idxs)
                return []

            # Step 1d: Crop/resize postprocessing (shared across all scenes)
            images, depths, extrinsics, intrinsics, view_infos = self._postprocess_views(raw_views, rng)

        else:
            # No 3D annotation: use shared sampling + simple resize
            # Expect row to have 'image_paths' or derive from img_dir
            # img_dir = row.get('img_dir', '')
            num_images = len(row['image_path'])

            # Discover available images in the directory
            all_image_paths = list(row['image_path'])

            if len(all_image_paths) == 0:
                return []

            # Reuse the same sampling strategy
            num_avail = len(all_image_paths)
            if num_avail != self.frame_num:
                idxs = self._sample_view_indices(num_avail, rng, max_distance=10)
            else:
                idxs = list(range(num_avail))
            images, depths, extrinsics, intrinsics, view_infos = self._simple_resize_views(
                all_image_paths, idxs, data_scene_name, this_scene
            )
            # Keep only sampled paths (must match dino_image count in pack_sequence)
            image_paths = [all_image_paths[i] for i in idxs]

        if len(images) == 0:
            return []

        # ── 2. Shuffle view order ──
        if self.shuffle_seq_views:
            indices = list(range(len(images)))
            self._rng.shuffle(indices)
            images = [images[i] for i in indices]
            depths = [depths[i] for i in indices]
            extrinsics = [extrinsics[i] for i in indices]
            intrinsics = [intrinsics[i] for i in indices]
            view_infos = [view_infos[i] for i in indices]
            assert len(image_paths) == len(images)
            image_paths = [image_paths[i] for i in indices]

        # ── 3. Build data dict using modular methods ──
        data = self._init_data()
        data['img_per_seq'] = self.frame_num
        # image_paths: for has_3d_annotation=True, use view_infos as paths;
        # for has_3d_annotation=False, image_paths was set from row['image_path'] sampled subset
        if len(image_paths) == 0:
            # Fallback: use view_infos as image identifiers
            image_paths = list(view_infos)
        data['image_paths'] = image_paths

        # Build template: text prompt + dino images
        text_with_images = '<dino_image>' * len(images)
        task = 'geo'
        answer = ''
        split_list = apply_template_qwenvl2_reconThenUnd(text_with_images, answer, task)

        # Count total dino images for split_start/split_end bi-directional attention
        total_dino_count = sum(1 for item in split_list if item['type'] == 'dino')
        dino_counter = 0

        # Prepare per-view metadata for _add_image
        images_queue = list(images)
        depths_queue = list(depths)
        extrinsics_queue = list(extrinsics)
        intrinsics_queue = list(intrinsics)
        view_infos_queue = list(view_infos)

        for item in split_list:
            try:
                if item['type'] == 'text':
                    data = self._add_text(data, item["value"], need_loss=item['loss'])
                elif item['type'] == 'dino':
                    image, images_queue = self.pop_first(images_queue)
                    depth_map, depths_queue = self.pop_first(depths_queue)
                    extri, extrinsics_queue = self.pop_first(extrinsics_queue)
                    intri, intrinsics_queue = self.pop_first(intrinsics_queue)
                    this_view_info, view_infos_queue = self.pop_first(view_infos_queue)

                    dino_meta = {
                        'scene_name': data_scene_name,
                        'depth_map': depth_map,
                        'extri_opencv': extri,
                        'intrinsic_': intri,
                    }

                    # Determine split_start/split_end for bi-directional attention
                    is_split_start = (dino_counter == 0)
                    is_split_end = (dino_counter == total_dino_count - 1)
                    dino_counter += 1

                    data = self._add_image(
                        data,
                        image,
                        dino_meta=dino_meta,
                        need_loss=False,
                        need_dino=True,
                        need_vit=False,
                        rng=rng,
                        view_info=this_view_info,
                        has_3d_annotation=has_3d_annotation,
                        split_start=is_split_start,
                        split_end=is_split_end,
                    )
            except AssertionError as e:
                print(e, 'skipping')
                return []

        return data
