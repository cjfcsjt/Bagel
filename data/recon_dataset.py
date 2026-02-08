from .distributed_iterable_dataset import DistributedIterableDataset

class SftJSONLIterableReconDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, dino_transform, tokenizer, frame_sampler,
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.dino_transform = dino_transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status

        self._rng = np.random.default_rng(shuffle_seed)
    
    def set_random_image_num(self, num):
        self.random_image_num = num
        self.frame_num = num
    
    def set_random_aspect_ratio(self, num):
        self.random_aspect_ratio = num
    
    def set_step_rng(self, rng):
        self._rng = np.random.default_rng(rng)
    
    def blender2opencv_c2w(self, pose):
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()

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
    
    def __iter__(self):
        # 分布式数据加载 checkpointing/resume 功能
        # 数据路径分配给不同的worker, 返回当前worker负责处理的数据路径列表和worker id
        # data_status记录每个worker已经处理到哪一行了
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}:"
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]

            allow_retry_times = 50
            retry_time = 0
            data_fail = False
            error = None
            pi3 = True
            if pi3:
                shuffle_seq_views = True
            
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                
                num_token = 0
                dino_image_tensor_list = []
                dino_thw = []
                dino_images = []
                text_ids_list = []
                sequence_plan = []

                images = []
                # 3d annotaions
                depths = []
                cam_points = []
                world_points = []
                point_masks = []
                extrinsics = []
                intrinsics = []

                view_infos = []
                original_sizes = []
                img_per_seq_list = []
                img_per_seq = self.frame_num
                
                # [image resolution] 是否启用高分辨率 则使用动态宽高比 否则固定1:1 正方形
                if high_resolution_training:
                    aspect_ratio = self.random_aspect_ratio
                else:
                    aspect_ratio = 1.0
                target_image_shape = self.get_target_shape(aspect_ratio)
                self.resolution = target_image_shape

                try:
                    data_item = json.loads(data)
                    if 'meta' in data_item:
                        data_meta = data_item['meta']
                    
                    data_scene_name = data_item['scene_name']
                    this_scene = data_item['seq_name']
                    text_ins = 'Reconstruct the 3D scene'
                    rng = self._rng # 这个值的状态会随着迭代，被外部改变

                    if data_scene_name == 'scannet':
                        pass
                    elif data_scene_name == 'blendmvs':
                        # BlendedMVS 数据集处理
                        # 数据格式: data_item 包含场景的元数据
                        scene_dir = data_item.get('scene_dir', None) # 场景根目录
                        if scene_dir is None:
                            raise ValueError(f"BlendedMVS scene_dir not found in data_item")
                        
                        # 获取所有可用的图像
                        cam_dir = os.path.join(scene_dir, 'cams')
                        all_view_names = sorted([f[:-8] for f in os.listdir(cam_dir) if not f.startwith('pair') and f.endswith('_cam.txt')])
                        num_imgs = len(all_view_names)

                        # 采样视图
                        if self.frame_num > 16 and rng.random() < self.random_sample_thres:
                            # 随机采样
                            should_replace = num_imgs < self.frame_num
                            idxs = list(rng.choice(range(num_imgs), size=self.frame_num, replace=should_replace))
                        else:
                            # 基于距离的采样策略
                            idxs = [rng.integers(0, num_imgs)]

                            blendedmvs_max_distance = 10 # BlendedMVS数据集的最大距离, 视图的数量较少

                            max_distance = int(blendedmvs_max_distance / 8 * self.frame_num)

                            start_idx = max(0, idxs[-1] - max_distance)
                            end_idx = min(num_imgs - 1, start_idx + 2 * max_distance)
                            start_idx = max(0, end_idx - 2 * max_distance)
                            valid_indices = np.arange(start_idx, end_idx + 1)
                        
                            if rng.random() < 0.5:
                                # 随机采样
                                should_replace = len(valid_indices) < self.frame_num - 1
                                idxs.extend(list(rng.choice(valid_indices, size=self.frame_num - 1, replace=should_replace)))
                            else:
                                # 分层采样
                                ref_frame_val = idxs[0]
                                num_additional_to_select = self.frame_num - 1
                                additional_selected_values = []
                                pool_for_others_values = list(valid_indices)
                                pool_for_others_values.sort()

                                should_replace_for_others = len(pool_for_others_values) < num_additional_to_select

                                if not pool_for_others_values:
                                    if should_replace_for_others:
                                        additional_selected_values = [ref_frame_val] * num_additional_to_select
                                else:
                                    if not should_replace_for_others and len(pool_for_others_values) >= num_additional_to_select:
                                        strata = np.array_split(pool_for_others_values, num_additional_to_select + 1)
                                        for stratum in strata:
                                            additional_selected_values.append(rng.choice(stratum))
                                    else:
                                        additional_selected_values = list(rng.choice(
                                            pool_for_others_values,
                                            num_additional_to_select,
                                            replace=(should_replace_for_others or (len(pool_for_others_values) < num_additional_to_select))
                                        ))

                                idxs = [ref_frame_val, *additional_selected_values]

                        # 加载每个视图的数据
                        for idx in idxs:
                            view_name = all_view_names[idx]
                            
                            # 加载相机内参和外参    
                            cam_file = os.path.join(cam_dir, f'{view_name}_cam.txt')
                            with open(cam_file, 'r') as f:
                                # 跳过第一行 (extrinsic tag)
                                f.readline()
                                # 读取外参矩阵 (4*4) - world2cam 格式
                                RT_world2cam = np.array([list(map(float, f.readline().split())) for _ in range(4)], dtype=np.float32)
                                # 转换为 cam2world 格式
                                extri_opencv = np.linalg.inv(RT_world2cam)

                                # 跳过空行和intrinsic标签
                                f.readline()
                                f.readline()
                                # 读取内参矩阵 (3*3)
                                intri_opencv = np.array([list(map(float, f.readline().split())) for _ in range(3)], dtype=np.float32)

                            # 加载 RGB 图像
                            image_path = os.path.join(scene_dir, "blended_images", f"{view_name}.jpg")
                            rgb_image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

                            # 加载深度图(PFM格式)
                            depth_path = os.path.join(scene_dir, "redered_depth_maps", f"{view_name}.pfm") 
                            depth_map = self._load_pfm_file(depth_path)

                            # 裁剪和调整尺寸
                            rgb_image, depth_map, intrinsic_ = self._crop_resize_if_necessary(
                                rgb_image, depth_map, intri_opencv.copy(), self.resolution, rng=rng, info=image_path)

                            images.append(rgb_image)
                            depths.append(depth_map)
                            extrinsics.append(extri_opencv.astype(np.float32))
                            intrinsics.append(intrinsic_.astype(np.float32))
                            view_infos.append(f"{data_scene_name}/{this_scene}/{view_name}")

                    ## 开始数据类型的检查和3d点云的生成 ##
                    # 打乱视图顺序可以增加训练的随机性，避免模型学习到固定的视图顺序模式
                    if shuffle_seq_views:
                        indices = list(range(len(images)))
                        self._rng.shuffle(indices)
                        # 同步打乱所有相关列表，保持视图的对应关系
                        images = [images[i] for i in indices]
                        depths = [depths[i] for i in indices]
                        extrinsics = [extrinsics[i] for i in indices]
                        intrinsics = [intrinsics[i] for i in indices]
                        view_infos = [view_infos[i] for i in indices]
                    
                    # 2. 将深度图转换为3d点云
                    # 为每个视图生成世界坐标系下的3d点云
                    
                    new_depths = [] # 清理后的深度图（无效点为0）
                    world_points = [] # 世界坐标系下的3d点
                    point_masks = [] # 有效点的mask
                    skip_this_scene = False # 标记是否跳过当前场景
                    for v, (img, depthmap, camera_pose, camera_intrinsics, view_info) in enumerate(zip(images, depths, extrinsics, intrinsics, view_infos)):
                        
                        width, height = img.size

                        assert np.isfinite(camera_pose).all(), f"NaN in camera pose for view {view_info}"

                        assert np.isfinite(depthmap).all(), f"NaN in depthmap for view {view_info}"

                        scene_label = view_info.split('/')[0]

                        # 根据数据集类型设置远距离裁剪阈值
                        # z_far 用于过滤掉过远的深度点，减少噪声
                        if scene_label in ['co3dv2', 'wildrgbd', 'blendedmvs']:
                            z_far = 0 # 不进行远距离裁剪（这些数据集距离较近）
                        elif scene_label in ['gtasfm', 'matrixcity', 'taskonomy', 'hypersim', 'nav_20w', 'vkitti', 'megadepth', 'dl3dv', 'omniworld', 'unreal4k']:
                            z_far = 0 # 这些数据集已经预处理过，不需要额外裁剪
                        elif scene_label in ['tartanair', 'scannet']:
                            z_far = 80 # 室内场景，裁剪80米以外的点
                        elif scene_label in ['scannetpp', 'arkitscenes']:
                            z_far = 120 # 较大的场景，裁剪120米以外的点
                        else:
                            z_far = 0 # 默认不裁剪
                        
                        # 将深度图转换为世界坐标系下的3d点云
                        # pts3d: (H, W, 3)
                        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                            depthmap, camera_intrinsics, camera_pose, z_far=z_far
                        )

                        # 进一步过滤无效点: 排除包含 NaN 或 Inf的点
                        valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
                        # 将无效点的深度设置为0
                        depthmap[~valid_mask] = 0.0

                        if not valid_mask.sum() > 0:
                            skip_this_scene = True
                            break
                        assert valid_mask.sum() > 0, f"No valid points in view {view_info}, depthmap: {depthmap}"

                        # 收集处理后的数据
                        new_depths.append(depthmap)
                        world_points.append(pts3d)
                        point_masks.append(valid_mask)

                    # 如果场景中所有视图都没有有效点，跳过该场景
                    if skip_this_scene:
                        print(f"Skipping scene {view_infos} due to no valid points")
                        continue
                    
                except Exception as e:
                    data_fail = True
                    retry_time += 1
                    error = e
                    print(
                        f"Failed to load scene {view_infos}, error: {e}", fluse=True
                    )
                    traceback.print_exc()
                    continue
                    
                # 3. 图像特征提取 (DINO Transform)
                raw_images = images
                transform_stride = self.patch_size

                for raw_image in raw_images:
                    # 应用DINO变换，提取视觉特征
                    image_tensor = self.dino_transform(raw_image, img_num=len(raw_images))
                    dino_images.append(raw_image)
                    dino_image_tensor_list.append(image_tensor)

                    height, width = image_tensor.shape[1:]
                    num_token += width * height // transform_stride ** 2

                    # spatial-temporal (t, h, w)
                    grid_t = 1
                    grid_h, grid_w = height // self.patch_size, width // self.patch_size
                    thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)
                    dino_thw.append(thw)

                # 4. 文本处理, to token IDs
                text_data = text_ins
                text_ids = self.tokenizer.encode(text_data)
                if len(text_ids) > 0:
                    text_ids_list.append(text_ids)
                    num_tokens += len(text_ids)
                    current_plan = {
                        'type': 'text',
                        'enable_cfg': 0, # 不启用 classifier-free guidance
                        'loss': 0, # 文本部分不计算loss
                        'special_token_loss': 0, # 特殊token不计算loss
                        'special_token_label': None, 
                    }
                    sequence_plan.append(current_plan)
                
                # 5. 为每个图像创建序列计划
                for _ in range(len(dino_image_tensor_list)):
                    current_plan = {
                        'type': 'dino_image',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                    sequence_plan.append(current_plan)
                
                if retry_tiem >= allow_retry_time:
                    raise error
                
                # 6. 生成批次数据
                yield dict(
                    # context
                    dino_image_tensor_list=dino_image_tensor_list,
                    dino_thw=dino_thw,
                    dino_images=dino_images,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    # 3d annotation
                    depths=new_depths,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    cam_points=cam_points,
                    world_points=world_points,
                    point_masks=point_masks,
                    # meta info
                    view_infos=view_infos,
                    image_per_seq=image_per_seq,
                    # 数据索引，用于断电续传
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )
        
        # 7 数据循环, 一轮结束后重新开始
        row_start_id = 0
        print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
        








