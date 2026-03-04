#!/usr/bin/env python3
import os
import os.path as osp
import re
import json
from tqdm import tqdm 
import numpy as np

# Add project root to sys.path to enable imports
import sys
project_root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from data.preprocessing.parallel import parallel_threads
import data.preprocessing.pi3.utils.cropping as cropping

# 导入用于3D点云生成的模块
import sys
from data.dataset_utils_vggt import save_ply_visualization_cpu_batch
from data.preprocessing.pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
from PIL import Image 
import torch


def _load_pose(path, ret_44=False):
    f = open(path)
    RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
    assert RT.shape == (4, 4)
    RT = np.linalg.inv(RT)  # world2cam to cam2world

    K = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
    assert K.shape == (3, 3)

    if ret_44:
        return K, RT
    return K, RT[:3, :3], RT[:3, 3]  # , depth_uint8_to_f32

def load_pfm_file(file_path):
    with open(file_path, "rb") as file:
        header = file.readline().decode("UTF-8").strip()

        if header == "PF":
            is_color = True
        elif header == "Pf":
            is_color = False
        else:
            raise ValueError("The provided file is not a valid PFM file.")

        dimensions = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
        if dimensions:
            img_width, img_height = map(int, dimensions.groups())
        else:
            raise ValueError("Invalid PFM header format.")

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

def load_crop_and_save(root, img, out_dir):
    """
    加载、裁剪并保存图像、深度图和相机参数

    Args:
        root (str): 图像和深度图的根目录
        img (str): 图像和深度图的文件名
        out_dir (str): 输出目录
    """
    if osp.isfile(osp.join(out_dir, img + ".npz")):
        return # 已经处理过，跳过
    
    # 1. 加载所有数据
    # 加载相机内参和外参 (世界坐标系到相机坐标系的变换)
    intrinsics_in, R_camin2world, t_camin2world = _load_pose(osp.join(root, "cams", img + "_cam.txt"))
    # 加载RGB图像
    color_image_in = cv2.cvtColor(
        cv2.imread(osp.join(root, "blended_images", img + ".jpg"), cv2.IMREAD_COLOR), 
        cv2.COLOR_BGR2RGB
    )
    # 加载深度图 (PFM格式)
    depth_image_in = load_pfm_file(osp.join(root, "rendered_depth_maps", img + ".pfm"))

    # 2. 裁剪图像和深度图
    H, W = color_image_in.shape[:2]
    assert H * 4 == W * 3 # 确保宽高比为4:3
    image, depthmap, intrinsics_out, R_in2out = _crop_image(
        intrinsics_in, color_image_in, depth_image_in, (512, 384)
    )


    # 3. 保存裁剪后的图像和深度图
    # 保存RGB图像 (JPEG格式, 质量为80)
    image.save(osp.join(out_dir, img + ".jpg"), quality=80)
    # 保存深度图 (OpenEXR格式, 保留浮点精度)
    cv2.imwrite(osp.join(out_dir, img + ".exr"), depthmap)

    # 4. 保存相机参数
    # 计算裁剪后的外参
    R_camout2world = R_camin2world @ R_in2out.T
    t_camout2world = t_camin2world
    np.savez(
        osp.join(out_dir, img + ".npz"),
        intrinsics=intrinsics_out, # 相机内参矩阵 (3*3)
        R_cam2world=R_camout2world, # cam2world 旋转矩阵 (3*3)
        t_cam2world=t_camout2world, # cam2world 平移向量 (3,)
    )

def _crop_image(intrinsics_in, color_image_in, depthmap_in, resolution_out=(800, 800)):
    """
    裁剪图像和深度图, 并更新相机内参
    Args:
        intrinsics_in (np.ndarray): 相机内参矩阵 (3*3)
        color_image_in (np.ndarray): RGB图像 (H*W*3)
        depthmap_in (np.ndarray): 深度图 (H*W)
        resolution_out (tuple): 输出分辨率 (H, W)
    Returns:
        image (np.ndarray): 裁剪后的RGB图像 (H*W*3)
        depthmap (np.ndarray): 裁剪后的深度图 (H*W)
        intrinsics_out (np.ndarray): 裁剪后的相机内参矩阵 (3*3)
        R_in2out (np.ndarray): 单位矩阵 因为只有缩放没有旋转
    """
    image, depthmap, intrinsics_out = cropping.rescale_image_depthmap(
        color_image_in, depthmap_in, intrinsics_in, resolution_out
    )
    R_in2out = np.eye(3) # 无旋转变换
    return image, depthmap, intrinsics_out, R_in2out


def save_scene_annotation(out_dir, scene_name):
    """
    为一个场景保存标注NPZ文件，包含所有视图的3D meta信息。
    文件按照排序后的图像名索引，使得可以通过 idx 直接访问。

    保存内容:
        - img_names: 排序后的图像名称列表 (N,)
        - intrinsics: 所有视图的内参矩阵 (N, 3, 3)
        - R_cam2world: 所有视图的cam2world旋转矩阵 (N, 3, 3)
        - t_cam2world: 所有视图的cam2world平移向量 (N, 3)
        - depth_files: 所有视图的深度图文件名 (N,)
        - rgb_files: 所有视图的RGB图像文件名 (N,)

    使用方式:
        data = np.load("scene_annotation.npz", allow_pickle=True)
        idx = 5
        K = data["intrinsics"][idx]          # (3, 3) 内参矩阵
        R = data["R_cam2world"][idx]          # (3, 3) 旋转矩阵
        t = data["t_cam2world"][idx]          # (3,) 平移向量
        rgb_file = str(data["rgb_files"][idx])  # RGB图像文件名
        depth_file = str(data["depth_files"][idx])  # 深度图文件名

    Args:
        out_dir: 处理后数据的输出目录 (包含 .jpg, .exr, .npz 文件)
        scene_name: 场景名称

    Returns:
        num_images: 该场景中的图像总数
    """
    # 找到所有已处理的 per-view .npz 文件（排除 scene_annotation.npz 本身）
    all_npz = sorted([
        f[:-4] for f in os.listdir(out_dir)
        if f.endswith(".npz") and f != "scene_annotation.npz"
    ])

    if len(all_npz) == 0:
        print(f"  [warn] no processed views found in {out_dir}")
        return 0

    # 收集所有视图的meta信息
    intrinsics_list = []
    R_cam2world_list = []
    t_cam2world_list = []
    rgb_files_list = []
    depth_files_list = []
    valid_img_names = []

    for img_name in all_npz:
        npz_path = osp.join(out_dir, img_name + ".npz")
        rgb_path = osp.join(out_dir, img_name + ".jpg")
        depth_path = osp.join(out_dir, img_name + ".exr")

        # 确保三件套都存在
        if not (osp.exists(npz_path) and osp.exists(rgb_path) and osp.exists(depth_path)):
            continue

        cam_params = np.load(npz_path)
        intrinsics_list.append(cam_params["intrinsics"])
        R_cam2world_list.append(cam_params["R_cam2world"])
        t_cam2world_list.append(cam_params["t_cam2world"])
        rgb_files_list.append(img_name + ".jpg")
        depth_files_list.append(img_name + ".exr")
        valid_img_names.append(img_name)

    if len(valid_img_names) == 0:
        print(f"  [warn] no complete views found in {out_dir}")
        return 0

    # 保存标注NPZ
    np.savez(
        osp.join(out_dir, "scene_annotation.npz"),
        img_names=np.array(valid_img_names),              # (N,)
        intrinsics=np.stack(intrinsics_list, axis=0),      # (N, 3, 3)
        R_cam2world=np.stack(R_cam2world_list, axis=0),    # (N, 3, 3)
        t_cam2world=np.stack(t_cam2world_list, axis=0),    # (N, 3)
        rgb_files=np.array(rgb_files_list),                # (N,)
        depth_files=np.array(depth_files_list),            # (N,)
    )

    print(f"  saved scene_annotation.npz with {len(valid_img_names)} views")
    return len(valid_img_names)


def generate_3d_visualization(root, out_dir, scene_name, num_views=10):
    """
    为指定场景生成 3D 点云可视化
    将深度图转换成3D 点云

    Args:
        root: 场景原始数据根目录
        out_dir: 输出目录
        scene_name: 场景名称
        num_views: 使用的视图数量
    """
    print(f"generating 3d visualization for {scene_name}")

    # 1. 获取所有可用的图像
    cam_dir = osp.join(root, "cams")
    all_imgs = sorted([f[:-8] for f in os.listdir(cam_dir) if not f.startswith("pair")])

    # 限制视图数量
    if len(all_imgs) > num_views:
        # 均匀采样
        indices = np.linspace(0, len(all_imgs) - 1, num_views, dtype=int)
        selected_imgs = [all_imgs[i] for i in indices]
    else:
        selected_imgs = all_imgs

    print(f"selected {len(selected_imgs)} images for 3d visualization")

    # 2. 准备数据容器
    images = []
    world_points = []
    point_masks = []
    view_infos = []

    # 3. 处理每个视图
    for img_name in tqdm(selected_imgs, desc=" processing images"):
        # 加载处理后的数据
        img_path = osp.join(out_dir, img_name + ".jpg")
        depth_path = osp.join(out_dir, img_name + ".exr")
        pose_path = osp.join(out_dir, img_name + ".npz")

        if not all([osp.exists(img_path), osp.exists(depth_path), osp.exists(pose_path)]):
            print(f"missing data for {img_name}")
            continue

        # 加载rgb图像
        rgb_image = np.array(Image.open(img_path))

        # 加载深度图
        depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depthmap.ndim == 3:
            depthmap = depthmap[:, :, 0] # 取第一个通道
        
        # 加载相机参数
        cam_params = np.load(pose_path)
        intrinsics = cam_params["intrinsics"]
        R_cam2world = cam_params["R_cam2world"]
        t_cam2world = cam_params["t_cam2world"]

        # 构建4x4的相机外参矩阵
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_cam2world
        camera_pose[:3, 3] = t_cam2world

        # 将深度图转换为3D点云(世界坐标系)
        # BlendMVS使用opencv的坐标系 与depthmap_to_absolute_camera_coordinates函数的坐标系一致
        # z_far=0 表示不进行距离裁剪 (BlendedMVS数据集通常距离较近)
        # 深度图中的深度值是沿着Z轴测量的
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
            depthmap, intrinsics, camera_pose, z_far=0
        )

        # 5. 验证点云数据的有效性
        valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
        depthmap[~valid_mask] = 0.0 

        if not valid_mask.sum() > 0:
            print(f"no valid points for {img_name}")
            continue
        
        # 6. 添加到数据容器
        images.append(rgb_image)
        world_points.append(pts3d)
        point_masks.append(valid_mask)
        view_infos.append(f"blendmvs/{scene_name}/{img_name}")

    if len(images) == 0:
        print(f"no valid images for {scene_name}")
        return
    
    # 7. 准备可视化字典
    predictions = {
        "world_points": world_points, # 所有视图的3D点 (世界坐标系)
        "point_masks": point_masks, # 所有视图的点云掩码
        "images": images, # 所有视图的RGB图像
        "view_infos": view_infos, # 所有视图的视图信息
    }

    # 8. 保存ply点云文件和图像网络
    print(f"saving ply file for {scene_name}")
    save_ply_visualization_cpu_batch(
        predictions, 
        scene_name,
        "QuickVis_BlendMVS",
        gt_only=True,
    )
    # 输出保存路径信息
    ply_output_dir = osp.join("save_path", "QuickVis_BlendMVS")
    print(f"done for {scene_name}")
    print(f"PLY files saved to: {osp.abspath(ply_output_dir)}")

def main(db_root, output_dir):
    # 列出所有场景序列
    sequences = [f for f in os.listdir(db_root) if len(f) == 24]
    assert sequences, f"found {len(sequences)} sequences in {db_root}"
    print(f"found {len(sequences)} sequences in {db_root}")

    # JSONL文件路径
    jsonl_path = osp.join(output_dir, "blendmvs_scenes.jsonl")
    jsonl_entries = []

    for i, seq in enumerate(tqdm(sequences)):
        out_dir = osp.join(output_dir, seq)
        os.makedirs(out_dir, exist_ok=True)

        # 生成裁剪之后的图像和深度图
        root = osp.join(db_root, seq)
        cam_dir = osp.join(root, "cams")
        # 遍历cam-dir目录下的所有文件名，跳过以pair开头的文件名，然后为每个文件名生成一个三元组，包含root, f[:-8], out_dir，放进列表func_args
        func_args = [
            (root, f[:-8], out_dir)
            for f in os.listdir(cam_dir)
            if not f.startswith("pair")
        ]
        # 多进程处理，对func_args中的每个三元组，调用load_crop_and_save函数
        parallel_threads(load_crop_and_save, func_args, star_args=True, leave=False)

        # 保存场景标注NPZ（汇总所有视图的meta信息，方便通过idx访问）
        num_images = save_scene_annotation(out_dir, seq)

        # 收集JSONL条目
        jsonl_entries.append({
            "scene_name": "blendmvs",
            "seq_name": seq,
            "num_images": num_images,
            "img_dir": osp.abspath(out_dir),
        })

        # 如果指定了可视化场景，那么可视化场景
        generate_3d_visualization(root, out_dir, seq, num_views=10)

    # 保存JSONL文件
    with open(jsonl_path, "w") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"JSONL index saved to: {osp.abspath(jsonl_path)}")
    print(f"Total scenes: {len(jsonl_entries)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main("/data/spatial_data/data/blendedmvs", "/data/spatial_data/data/blendedmvs/processed")



