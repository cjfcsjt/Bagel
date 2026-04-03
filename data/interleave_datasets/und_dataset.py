import io
import random
import os 
from PIL import Image, ImageFile, PngImagePlugin
from idna import intranges_contain

from .interleave_t2i_dataset_vae import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb, apply_template_qwenvl2, apply_template_qwenvl2_reconThenUnd
from ..distributed_iterable_dataset import DistributedIterableDataset
from ..dataset_utils_vggt import *
import torch
from copy import deepcopy
import traceback
import cv2
import modeling.pi3.utils.cropping as cropping
from modeling.pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
from .draw_marker import DRAW_FUNCTIONS
import torch.distributed as dist
import json
Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class UndIterableDataset(ParquetStandardIterableDataset, DistributedIterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.img_size = 224  # 518
        self.patch_size = 14
        self.aug_scale = [0.8, 1.2] 
        self.rescale = True
        self.rescale_aug = True
        self.landscape_check = False  # True
        self.training = True  # Consider making this configurable
        # self._rng = np.random.default_rng(self.shuffle_seed)
        self.random_sample_thres = 0.1 
        self.aug_crop = 0 
        self.aug_focal = None 
        self.z_far = 0  
        self.frame_num = 8 ###harcode 
        self.random_aspect_ratio = 1.0
        self.resolution = [518, 518]  
        self.spar5m_get_nearby_ids = False
        self.shuffle_seq_views = True

    def pop_first(self, arr):
        first_val = arr[0]
        new_arr = arr[1:]
        return first_val, new_arr
    def set_random_image_num(self, num):
        self.random_image_num = num     
        self.frame_num = num
    def set_random_aspect_ratio(self, num):
        self.random_aspect_ratio = num     
    def set_step_rng(self, rng):
        self._rng = np.random.default_rng(rng)

    def draw_image(self, image, data_item):
        draw_fn = DRAW_FUNCTIONS.get(data_item.get('type', None))
        if draw_fn is None:
            print(data_item)
            raise ValueError(f"Unsupported data type: {data_item.get('type', None)}")
        draw_fn(image, data_item)

    # ── 需要在图片上绘制标注的 spar 单图任务类型 ──
    _SPAR_SINGLE_IMAGE_TYPES = {
        "obj_spatial_relation_oo",
        "depth_prediction_oc",
        "depth_prediction_oo",
        "distance_prediction_oc",
        "distance_prediction_oo",
        "distance_infer_center_oc",
        "distance_infer_center_oo",
        "spatial_volume_infer",
        "spatial_imagination_oc",
        "spatial_imagination_oo",
    }

    # ── 需要在图片上绘制标注的 spar 多图/视频任务类型 ──
    _SPAR_MULTI_IMAGE_TYPES = {
        "position_matching",
        "view_change_infer",
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
        "obj_spatial_relation_oc_mv",
        "obj_spatial_relation_oo_mv",
        "distance_infer_center_oc_mv",
        "distance_infer_center_oo_mv",
        "spatial_imagination_oc_mv",
        "spatial_imagination_oo_mv",
        "spatial_imagination_map_mv",
        "camera_motion_infer",
        "distance_prediction_oo_video",
        "distance_infer_center_oo_video",
        "spatial_imagination_oo_video",
        "spatial_imagination_oc_video",
        "spatial_imagination_oc_video_hard",
        "spatial_imagination_oo_video_hard",
        "obj_frame_locate",
        "appearance_order",
        "room_size",
        "obj_count",
        "nav",
    }

    def _load_vit_images(self, dataset_name, all_image_paths, row):
        """
        从 image_path 加载 VIT 分支所需的 PIL 图片列表。

        根据数据集类型决定是否需要在图片上绘制标注（如 spar 数据集的
        bounding-box / marker 等）。非 spar 数据集直接从文件路径读取为 RGB 图片。

        Args:
            dataset_name: 数据集名称，用于判断是否为 spar 系列。
            all_image_paths: list[str]，图片文件路径。
            row: 当前行数据，用于读取 metadata。

        Returns:
            list[PIL.Image.Image]: 加载（并可能绘制了标注）后的 RGB 图片列表。
        """
        if 'spar' not in dataset_name:
            # 非 spar 数据集：直接从文件路径读取
            return [pil_img2rgb(Image.open(p)) for p in all_image_paths]

        # ── spar 数据集：需要根据 metadata 绘制标注 ──
        try:
            metadata = row['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            metadata = metadata['metadata']['spar_info']
            # spar_info 可能仍是字符串（旧 Parquet 文件），兼容处理
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            task_type = metadata.get('type', None)
            images = []
            if len(all_image_paths) == 1:
                assert task_type in self._SPAR_SINGLE_IMAGE_TYPES, f"Expected single-image task type, but got {task_type}"
                try:
                    image = Image.open(all_image_paths[0]).convert("RGB")
                    self.draw_image(image, metadata)
                    images.append(image)
                except Exception as e:
                    print(f"[WARN] draw_image failed | img_path={all_image_paths[0]} "
                            f"| data_item_id={metadata.get('id', 'unk')} | error={e}")
                return images
            elif len(all_image_paths) > 1:
                assert task_type in self._SPAR_MULTI_IMAGE_TYPES, f"Expected multi-image task type, but got {task_type}"
                images = [Image.open(p).convert('RGB') for p in all_image_paths]
                self.draw_image(images, metadata)   
                return images
            else:
                raise ValueError(f"No images found in all_image_paths for task_type={task_type}")
            
        except Exception as e:
            print(f'[WARN] _load_vit_images spar fallback: {e}')
            print(f'  metadata={row.get("metadata", "N/A")}')
            return [pil_img2rgb(Image.open(p)) for p in all_image_paths]

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # NOTE: Here we don't care about portrait image.
        # assert resolution[0] >= resolution[1]
        # if H > 1.1*W:
        #     # image is portrait mode
        #     resolution = resolution[::-1]
        # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        #     # image is square, so we chose (portrait, landscape) randomly
        #     if rng.integers(2):
        #         resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other

    # ────────────────────────── View Sampling ──────────────────────────

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

                # Fill missing samples if stratified sampling produced fewer than needed
                while len(additional_selected_values) < num_additional_to_select:
                    additional_selected_values.append(rng.choice(pool_for_others_values))

                idxs = [ref_frame_val, *additional_selected_values]

        assert len(idxs) == self.frame_num, (
            f"Expected {self.frame_num} frames, but got {len(idxs)} frames. idxs: {idxs}"
        )
        return idxs

    # ────────────────────────── Depth & Camera Loading ──────────────────────────

    def _load_depth_and_camera(self, scene_name, depth_path, pose, intri, image):
        """
        Load depth map and camera parameters based on scene type.
        Extracts the per-scene depth loading logic from the old _add_image.

        Args:
            scene_name: str, dataset scene name
            depth_path: str, path to depth file
            pose: array-like, camera extrinsic (will be reshaped to 4x4)
            intri: array-like, camera intrinsic (will be reshaped and sliced to 3x3)
            image: np.ndarray, RGB image (H, W, 3)

        Returns:
            depth_map: np.ndarray (H, W) float
            extri_opencv: np.ndarray (4, 4) float
            intri_opencv: np.ndarray (3, 3) float
            image: np.ndarray (possibly resized to match depth)
        """
        if scene_name == 'matterport3d':
            with Image.open(depth_path) as depth_img:
                depth_map = np.array(depth_img).astype(np.int32) / 4000.0
            depth_map[~np.isfinite(depth_map)] = 0

            threshold = (
                np.percentile(depth_map[depth_map > 0], 98)
                if depth_map[depth_map > 0].size > 0
                else 0
            )
            depth_map[depth_map > threshold] = 0.0

            extri_opencv = np.array(pose).reshape((4, 4))
            intri_opencv = np.array(intri)[:3, :3]

        elif scene_name == 'scannet':
            with Image.open(depth_path) as depth_img:
                depth_map = np.array(depth_img).astype(np.int32) / 1000.0
            depth_map[~np.isfinite(depth_map)] = 0
            if depth_map.shape[0] != image.shape[0]:
                image = cv2.resize(image, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_LINEAR)
            extri_opencv = np.array(pose).reshape((4, 4))
            intri_opencv = np.array(intri)[:3, :3]

        elif scene_name == '3rscan':
            with Image.open(depth_path) as depth_img:
                depth_map = np.array(depth_img).astype(np.int32) / 1000.0
            depth_map[~np.isfinite(depth_map)] = 0
            extri_opencv = np.array(pose).reshape((4, 4))
            intri_opencv = np.array(intri)[:3, :3]

        elif scene_name == 'scannetpp':
            with Image.open(depth_path) as depth_img:
                depth_map = np.array(depth_img).astype(np.int32) / 1000.0
            depth_map[~np.isfinite(depth_map)] = 0
            if depth_map.shape[0] != image.shape[0] or depth_map.shape[1] != image.shape[1]:
                depth_map = cv2.resize(
                    depth_map,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            extri_opencv = np.array(pose).reshape((4, 4))
            intri_opencv = np.array(intri)[:3, :3]

        elif scene_name == 'structured3d':
            with Image.open(depth_path) as depth_img:
                depth_map = np.array(depth_img).astype(np.int32) / 1000.0
            depth_map[~np.isfinite(depth_map)] = 0
            extri_opencv = np.array(pose).reshape((4, 4))
            extri_opencv[:3, 3] = extri_opencv[:3, 3] / 1000.0
            intri_opencv = np.array(intri)[:3, :3]

        else:
            raise ValueError(f"Unsupported scene type for depth loading: {scene_name}")

        return depth_map, extri_opencv, intri_opencv, image

    # ────────────────────────── Postprocessing (Shared) ──────────────────────────

    def _postprocess_views(self, raw_views, rng):
        """
        Apply crop/resize preprocessing to raw views that have 3D annotations.

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

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.
        
        Args:
            aspect_ratio: Target aspect ratio
            
        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape


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
            'repeat_mask': [],
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': 0,  #int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, dino_meta, need_loss, need_dino, need_vit, enable_cfg=True, rng=None, view_info=None, has_3d_annotation=True, split_start=True, split_end=True):
        assert need_loss or need_dino or need_vit

        if need_dino:
            if has_3d_annotation:
                # For refactored flow, 3D data is already loaded and preprocessed
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
                if scene_label in ['scannet']:
                    z_far = 80
                elif scene_label in ['matterport3d']:
                    z_far = 80
                elif scene_label in ['3rscan']:
                    z_far = 80
                else:
                    z_far = 80

                assert np.isfinite(extri_opencv).all(), f'NaN in camera pose for view {view_info}'
                assert np.isfinite(depth_map).all(), f'NaN in depthmap for view {view_info}'

                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depth_map, intrinsic_, extri_opencv, z_far=z_far)
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
                # Simple resize to target resolution
                # image = image.resize((self.resolution[1], self.resolution[0]), PIL.Image.LANCZOS)
                data['dino_images'].append(image)
                data['view_infos'].append(view_info)
                data['original_sizes'].append(np.array(self.resolution))
                
                # Placeholder data
                patch_h = self.resolution[0] // self.patch_size
                patch_w = self.resolution[1] // self.patch_size
                data['depths'].append(np.zeros((self.resolution[0], self.resolution[1]), dtype=np.float32))
                data['extrinsics'].append(np.eye(4, dtype=np.float32))
                data['intrinsics'].append(np.eye(3, dtype=np.float32))
                data['new_depths'].append(np.zeros((self.resolution[0], self.resolution[1]), dtype=np.float32))
                data['world_points'].append(np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.float32))
                data['point_masks'].append(np.zeros((self.resolution[0], self.resolution[1]), dtype=bool))
            
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

            vit_image_tensor = self.vit_transform(image, img_num=1) 
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor.clone())
            # data['image_grid_thw_list'].append(image_grid_thw[0])

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True): 
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data

    def _parse_row_video3dllm(self, row):
        """
        处理 video3dllm 数据集（VEGA-3D 五个数据集）。

        Parquet 行中已包含离线预计算的 3D 数据：
          - image_path:   所有帧的 RGB 图片路径
          - depth_list:   所有帧的深度图路径
          - poses:        所有帧的 cam2world 位姿（已 axis_align）
          - depth_intrinsic: 深度内参
          - world_coords: JSON 序列化的 (V, H, W, 3) 世界坐标
          - boundary:     [x_min, x_max, y_min, y_max, z_min, z_max]
          - objects:      JSON 序列化的物体框列表

        在线处理：
          1. 帧采样（_sample_view_indices）：从所有帧中采样 frame_num 帧
          2. 从预计算的 world_coords 中取对应帧的子集
          3. 加载采样帧的 RGB 图片（VIT 分支）
          4. 构建 sequence_plan（仅 VIT 分支，与普通 und 数据集一致）

        注意：question 中的 <image> token 数量固定为 1（VEGA-3D 格式），
        而 VIT 分支加载的是采样后的多帧图片，因此需要将 question 中的
        单个 <image> 替换为 frame_num 个 <image>。
        """
        question = row["question"]
        answer = row["answer"]
        dataset_name = row['dataset_name']
        data_scene_name = row['scene_name']

        try:
            # ── Step 1: 读取离线预计算的 3D 数据 ──
            all_image_paths = list(row['image_path'])
            num_imgs = len(all_image_paths)

            # 读取预计算的世界坐标
            world_coords_json = row.get('world_coords', None)
            boundary = row.get('boundary', None)
            objects_json = row.get('objects', None)

            has_3d = (world_coords_json is not None and world_coords_json != '')

            # ── Step 2: 在线帧采样 ──
            rng = np.random.default_rng(
                abs(hash(row.get('metadata', '') + str(num_imgs))) % (2**31)
            )

            if num_imgs <= self.frame_num:
                # 帧数不足，全部使用
                idxs = list(range(num_imgs))
            else:
                max_distance = 20 if data_scene_name in ['scannet'] else 10
                idxs = self._sample_view_indices(num_imgs, rng, max_distance=max_distance)

            # ── Step 3: 取采样帧的图片路径 ──
            sampled_image_paths = [all_image_paths[i] for i in idxs]
            n_sampled = len(sampled_image_paths)

            # ── Step 4: 从预计算的 world_coords 中取对应帧 ──
            if has_3d:
                try:
                    world_coords_all = np.array(
                        json.loads(world_coords_json), dtype=np.float32
                    )  # (V_all, H, W, 3)
                    world_coords_sampled = world_coords_all[idxs]  # (n_sampled, H, W, 3)
                except Exception as e:
                    print(f"[WARN] video3dllm: failed to parse world_coords: {e}")
                    has_3d = False
                    world_coords_sampled = None
            else:
                world_coords_sampled = None

            # ── Step 5: 加载采样帧的 RGB 图片（VIT 分支） ──
            raw_images = []
            for img_path in sampled_image_paths:
                try:
                    raw_images.append(pil_img2rgb(Image.open(img_path)))
                except Exception as e:
                    print(f"[WARN] video3dllm: failed to load image {img_path}: {e}")
                    return []

            if not raw_images:
                return []

            # ── Step 6: 构建 question（将单个 <image> 替换为 n_sampled 个 <image>） ──
            # VEGA-3D 的 question 格式：以单个 <image> 开头
            # 我们需要将其替换为 n_sampled 个 <image>
            if '<image>' in question:
                # 替换第一个 <image> 为 n_sampled 个 <image>
                question_with_frames = question.replace(
                    '<image>',
                    '<image>' * n_sampled,
                    1  # 只替换第一个
                )
            else:
                # 没有 <image> token，在开头添加
                question_with_frames = '<image>' * n_sampled + '\n' + question

            # ── Step 7: 构建 data dict ──
            data = self._init_data()
            data['img_per_seq'] = n_sampled
            data['image_paths'] = sampled_image_paths

            # 将 <image> 替换为 <vit_image>
            text_with_images = question_with_frames.replace('<image>', '<vit_image>')
            split_list = apply_template_qwenvl2_reconThenUnd(text_with_images, answer, task='und')

            raw_images_queue = list(raw_images)

            for item in split_list:
                try:
                    if item['type'] == 'text':
                        data = self._add_text(data, item["value"], need_loss=item['loss'])
                    elif item['type'] == 'vit':
                        image, raw_images_queue = self.pop_first(raw_images_queue)
                        dino_meta = {'scene_name': data_scene_name}
                        data = self._add_image(
                            data,
                            image,
                            dino_meta=dino_meta,
                            need_loss=False,
                            need_dino=False,
                            need_vit=True,
                        )
                except AssertionError as e:
                    print(e, 'skipping video3dllm row')
                    return []

            # ── Step 8: 附加 3D 信息到 data（供下游使用） ──
            if has_3d and world_coords_sampled is not None:
                data['video3dllm_world_coords'] = world_coords_sampled  # (n_sampled, H, W, 3)
            if boundary is not None:
                data['video3dllm_boundary'] = list(boundary)
            if objects_json is not None and objects_json != '':
                try:
                    data['video3dllm_objects'] = json.loads(objects_json)
                except Exception:
                    data['video3dllm_objects'] = []

            return data

        except Exception as e:
            print(f"[WARN] _parse_row_video3dllm failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def parse_row(self, row):
        question = row["question"]
        answer = row["answer"]

        data_scene_name = row['scene_name']
        dataset_name = row['dataset_name']

        # ── video3dllm 数据集：使用离线预计算的 3D 数据 + 在线帧采样 ──
        if dataset_name.startswith('video3dllm'):
            return self._parse_row_video3dllm(row)

        # ── Step 0: Extract scene metadata (pure metadata, no I/O) ──
        has_3d_annotation = (
            'depth_list' in row and row['depth_list'] is not None and len(row.get('depth_list', [])) > 0 and
            'poses' in row and row['poses'] is not None and len(row.get('poses', [])) > 0
        )

        all_image_paths = list(row['image_path'])
        this_scene = row['scene_name']
        num_imgs = len(all_image_paths)

        if has_3d_annotation:
            all_depth_paths = list(row['depth_list'])
            all_poses = list(row['poses'])
            assert len(all_image_paths) == len(all_depth_paths) == len(all_poses), \
                f"Mismatch: images={len(all_image_paths)}, depths={len(all_depth_paths)}, poses={len(all_poses)}"
            # Intrinsic: per-scene shared value, pick once (not per-view append)
            if data_scene_name in ['scannet', 'structured3d']:
                scene_intrinsic = row['depth_intrinsic']
            else:
                scene_intrinsic = row['intrinsic']

        # ── Step 1: Sample view indices (shared across all scenes) ──
        # if num_imgs != self.frame_num:
        #     max_distance = 20 if data_scene_name in ['scannet'] else 10
        #     idxs = self._sample_view_indices(num_imgs, rng, max_distance=max_distance)
        # else:
        #     idxs = list(range(num_imgs))

        # Apply sampling — only for dino branch (frame_num aligned)
        # dino_image_paths = [all_image_paths[i] for i in idxs]
        # dino_view_infos_raw = [f'{data_scene_name}/{dataset_name}/{this_scene}/{i}' for i in idxs]

        # ── Step 2: Load raw data + postprocess for dino branch ──
        # if has_3d_annotation:
        #     raw_views = []
        #     for j, idx in enumerate(idxs):
        #         rgb_image = np.array(Image.open(dino_image_paths[j]).convert("RGB"))
        #         depth_path = all_depth_paths[idx]
        #         pose = np.vstack(all_poses[idx]).reshape((4, 4))
        #         intri = np.vstack(scene_intrinsic).reshape((4, 4))

        #         depth_map, extri_opencv, intri_opencv, rgb_image = self._load_depth_and_camera(
        #             data_scene_name, depth_path, pose, intri, rgb_image
        #         )
        #         raw_views.append({
        #             'image': rgb_image,
        #             'depth': depth_map,
        #             'extrinsic': extri_opencv,
        #             'intrinsic': intri_opencv,
        #             'view_info': dino_view_infos_raw[j],
        #         })

        #     # Unified crop/resize postprocessing
        #     dino_images, dino_depths, dino_extrinsics, dino_intrinsics, dino_view_infos = \
        #         self._postprocess_views(raw_views, rng)
        # else:
        #     # No 3D annotation: simple resize
        #     dino_images, dino_depths, dino_extrinsics, dino_intrinsics, dino_view_infos = \
        #         self._simple_resize_views(dino_image_paths, list(range(len(dino_image_paths))), data_scene_name, this_scene)

        # assert len(dino_images) > 0, f"No valid dino images found for scene {this_scene}"

        # ── Step 3: Load VIT images (根据数据集类型决定是否绘制标注) ──
        raw_images = self._load_vit_images(dataset_name, all_image_paths, row)

        # ── Step 4: Shuffle view order (dino branch only, vit is independent) ──
        # if self.shuffle_seq_views:
        #     indices = list(range(len(dino_images)))
        #     self._rng.shuffle(indices)
        #     dino_images = [dino_images[i] for i in indices]
        #     dino_depths = [dino_depths[i] for i in indices]
        #     dino_extrinsics = [dino_extrinsics[i] for i in indices]
        #     dino_intrinsics = [dino_intrinsics[i] for i in indices]
        #     dino_view_infos = [dino_view_infos[i] for i in indices]
        #     dino_image_paths = [dino_image_paths[i] for i in indices]

        # ── Step 5: Build data dict ──
        data = self._init_data()
        data['img_per_seq'] = len(raw_images)
        data['image_paths'] = all_image_paths

        # Build template: dino images + question with vit images
        # question = '<image>' * len(raw_images) + question
        # question = question.replace('<image>\n', '<image>')
        num_vit_tokens = question.count('<image>')
        assert num_vit_tokens == len(raw_images), f"num_vit_tokens={num_vit_tokens}, len(raw_images)={len(raw_images)}"

        text_with_images = question.replace('<image>', '<vit_image>')
        # text_with_images = '<dino_image>' * len(dino_images) + text_with_images
        split_list = apply_template_qwenvl2_reconThenUnd(text_with_images, answer, task='und')

        # Count total dino images for split_start/split_end bi-directional attention
        # total_dino_count = sum(1 for item in split_list if item['type'] == 'dino')
        # dino_counter = 0

        # Prepare per-view queues for _add_image
        # dino_images_queue = list(dino_images)
        # dino_depths_queue = list(dino_depths)
        # dino_extrinsics_queue = list(dino_extrinsics)
        # dino_intrinsics_queue = list(dino_intrinsics)
        # dino_view_infos_queue = list(dino_view_infos)

        for item in split_list:
            try:
                if item['type'] == 'text':
                    data = self._add_text(data, item["value"], need_loss=item['loss'])
                # elif item['type'] == 'dino':
                #     image, dino_images_queue = self.pop_first(dino_images_queue)
                #     depth_map, dino_depths_queue = self.pop_first(dino_depths_queue)
                #     extri, dino_extrinsics_queue = self.pop_first(dino_extrinsics_queue)
                #     intri, dino_intrinsics_queue = self.pop_first(dino_intrinsics_queue)
                #     this_view_info, dino_view_infos_queue = self.pop_first(dino_view_infos_queue)

                #     dino_meta = {
                #         'scene_name': data_scene_name,
                #         'depth_map': depth_map,
                #         'extri_opencv': extri,
                #         'intrinsic_': intri,
                #     }

                #     # Determine split_start/split_end for bi-directional attention across all dino images
                #     is_split_start = (dino_counter == 0)
                #     is_split_end = (dino_counter == total_dino_count - 1)
                #     dino_counter += 1
                #     data = self._add_image(
                #         data, 
                #         image,
                #         dino_meta=dino_meta,
                #         need_loss=False, 
                #         need_dino=True, 
                #         need_vit=False, 
                #         rng=rng,
                #         view_info=this_view_info,
                #         has_3d_annotation=has_3d_annotation,
                #         split_start=is_split_start,
                #         split_end=is_split_end,
                #     )
                elif item['type'] == 'vit':
                    image, raw_images = self.pop_first(raw_images)
                    dino_meta = {'scene_name': data_scene_name}
                    data = self._add_image(
                            data, 
                            image,
                            dino_meta=dino_meta,
                            need_loss=False, 
                            need_dino=False, 
                            need_vit=True, 
                        )
            except AssertionError as e:
                print(e, 'skipping')
                return [] 

        return data
