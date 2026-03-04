import os
import random
random.seed(224)
import sys
import glob
import time
import threading
from typing import List, Optional
import torch.nn.functional as F
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torchvision
from g2vlm_utils import load_model_and_tokenizer, save_ply_visualization
from eval.point.scannet.dataset import ScanNetSequenceDataset
from eval.point.utils import calculate_gt_tracks, visualize_comparison_tracks, visualize_gt_tracks_step_by_step, compute_errors_for_sample
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Config:
    """简单的配置类，替代 hydra"""
    def __init__(self):
        # 模型配置
        self.model_path = "/data/spatial_data/ckpt/0000200"
        self.save_path = "results/scannet_test_1500_results.ply"
        self.vit_path = "/data/spatial_data/hf/qwen2-vl-2b"
        
        # 数据集配置
        self.image_folder = "/data/spatial_data/data/scannet/scannet_test_1500"
        self.num_views = 8
        self.image_size = (480, 640)
        self.max_frame_skip = 35
        
        # 评估配置
        self.random_seed = 8
        self.scale_factor = 0.25
        self.num_corr = 10
        self.size = 512

def main():
    config = Config()
    model_name = config.model_path.split('/')[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # 创建保存目录
    os.makedirs(config.save_path, exist_ok=True)
    
    # 加载模型
    print("Loading model...")
    model, tokenizer, new_token_ids, vit_image_transform, dino_transform = load_model_and_tokenizer(config)
    model = model.to(device)
    model.eval()

    # 读取数据集
    # images = ['/path/to/image1.jpg', '/path/to/image2.jpg', '/path/to/image3.jpg']
    # 创建数据集和 DataLoader
    print("Loading dataset...")
    dataset = ScanNetSequenceDataset(
        root_dir=config.image_folder,
        # num_views=config.num_views,
        # image_size=tuple(config.image_size),
        # max_frame_skip=35,
        # split="valid",
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    
    # 评估指标
    dataset_2d_errors = []
    dataset_3d_errors = []

    # max_samples = args.max_samples if args.max_samples > 0 else len(loader)

    # 第0帧起始点 ──────> attention map ──────> 光流 ──────> 第v帧预测位置
    #  (x, y)           (跨视图关联)      (dx, dy)      (x+dx, y+dy)
    for idx, batch in enumerate(tqdm(loader, desc="Evaluating tracking")):
        # if idx >= max_samples:
            # break
            
        # 获取数据
        # DataLoader collate 会把 ['p1', 'p2', ...] 变成 [['p1'], ['p2'], ...]
        # 需要展平回 ['p1', 'p2', ...]
        image_paths = batch["image_paths"]
        images_no_norm = batch["images"][0].cuda(non_blocking=True)
        if isinstance(image_paths[0], (list, tuple)):
            image_paths = [p[0] for p in image_paths]  # 展平: [['p1'], ['p2']] -> ['p1', 'p2']
        
        depths = batch["depth"][0].cuda(non_blocking=True)  # (V, 1, H, W)
        intrinsics = batch["intrinsics"][0].cuda(non_blocking=True)  # (3, 3) or (V, 3, 3)
        Rts = batch["Rt"][0].cuda(non_blocking=True)  # (V, 4, 4)
        masks = batch["mask"][0].cuda(non_blocking=True)  # (V, 1, H, W)
        
        # V, C, H, W = images.shape
        V = 8
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).repeat(V, 1, 1)
        
        # 采样起始点
        num_tracks = config.num_corr
        masked_coords = masks[0].squeeze().nonzero(as_tuple=False)
        
        if len(masked_coords) == 0:
            print(f"Warning: No valid mask found for sample {idx}. Skipping.")
            continue
        if len(masked_coords) < num_tracks:
            print(f"Warning: Not enough points in mask for sample {idx}. Using {len(masked_coords)} points.")
            num_tracks = len(masked_coords)
        
        set_all_seeds(42 + idx)
        indices = torch.randperm(len(masked_coords))[:num_tracks]
        start_points = masked_coords[indices][:, [1, 0]].float()  # (num_tracks, 2), (x, y) 格式
        
        gt_tracks = calculate_gt_tracks(
            start_points,
            depths,
            intrinsics,
            Rts,
            source_inverse=True,
            target_inverse=False,
            record_all_projections=False,
            check_start_in_bounds=True,
            invalidate_if_never_visible=True,
        )
        gt_step_dir = f"viz/scannet_tracks_{num_tracks}/{model_name}/gt_steps_{idx}"
        nrow = 4
        # visualize_gt_tracks_step_by_step(
        #     [img for img in images_no_norm], gt_tracks, gt_step_dir, nrow=nrow
        # )
        # pred_tracks = torch.zeros((num_tracks, V, 2), device="cuda")
        # pred_tracks[:, 0, :] = start_points
        
        # 模型预测
        try:
            with torch.no_grad():
                tracking_result, _ = model.track_points(
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    dino_image_transform=dino_transform,
                    images=image_paths,  # 使用处理后的路径列表
                    start_points=start_points,
                    use_attention_maps=True,
                    source_view_idx=0,
                    output_hw=(images_no_norm.shape[-2], images_no_norm.shape[-1]),  # gt 分辨率 (H, W)
                )
            
            pred_tracks = tracking_result['pred_tracks']  # (num_tracks, V, 2)
            flow_maps = tracking_result['flow_maps']      # (V-1, 2, H, W)

            print(f"Tracked {num_tracks} points across {pred_tracks.shape[1]} views")
            
        except Exception as e:
            print(f"Error tracking points for sample {idx}: {e}")
            continue
        
        save_path = (
            f"viz/scannet_tracks_{num_tracks}/{model_name}/comparison_{idx}.png"
        )
        visualize_comparison_tracks(
            [img for img in images_no_norm], pred_tracks, gt_tracks, save_path, nrow=nrow
        )

        errors_2d_sample, errors_3d_sample = compute_errors_for_sample(
            pred_tracks, gt_tracks, depths, intrinsics
        )
        if errors_2d_sample.numel() > 0:
            dataset_2d_errors.append(errors_2d_sample.cpu())
            dataset_3d_errors.append(errors_3d_sample.cpu())


def unproject_point(x, y, depth, intrinsic):
    """将2D点反投影到3D空间"""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    z = depth
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    
    return torch.stack([x_3d, y_3d, z])
    
if __name__ == "__main__":
    main()