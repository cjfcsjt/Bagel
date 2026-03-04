import os
import random
random.seed(224)
import sys
import glob
import time
import threading
import argparse
from typing import List, Optional
import torch.nn.functional as F
import numpy as np
import torch
from tqdm.auto import tqdm
import torchvision
from g2vlm_utils import load_model_and_tokenizer, save_ply_visualization

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description="Demo for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/dl3dv/", help="Path to folder containing images"
)
# parser.add_argument(
#     "--image_folder", type=str, default="examples/arkitscenes/", help="Path to folder containing images"
# )
parser.add_argument("--model_path",type=str, default="InternRobotics/G2VLM-2B-MoT")
parser.add_argument("--save_path",type=str, default="results/arkitscenes_results.ply")


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # 读取数据集
    # images = ['/path/to/image1.jpg', '/path/to/image2.jpg', '/path/to/image3.jpg']
    image_folder = args.image_folder
    image_names = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    print(image_names)
    # 读取3d gt标注
    
    # 数据预处理

    # 在第一帧采样起始点
    # 获取mask内有效像素坐标, 随机选num_tracks个, (y,x) -> (x,y) 即 (u,v)

    # 初始化轨迹张量
    # 第一帧的轨迹位置直接设为采样的起始点，后续视图的位置由模型预测填充 pred_tracks 形状为 (num_tracks, V, 2)
    # num_tracks: 要追踪的点的数量，即在第一帧图像上采样多少个点，然后预测这些点在后续视图中的位置。
    # V: 视图数量
    # 2: (x, y) 坐标
    # 初始化预测轨迹张量，形状为(追踪点数, 视图数, 2)
    # pred_tracks = torch.zeros((num_tracks, V, 2), device="cuda")

    # 第一帧的轨迹位置就是采样的起始点本身
    # pred_tracks[:, 0, :] = start_points
    
    # 读取data_loader
    model, tokenizer, new_token_ids , vit_image_transform, dino_transform = load_model_and_tokenizer(args)
    # 第0帧起始点 ──────> attention map ──────> 光流 ──────> 第v帧预测位置
    #  (x, y)           (跨视图关联)      (dx, dy)      (x+dx, y+dy)
    # pred = model.recon(
    #     tokenizer,
    #     new_token_ids,
    #     dino_transform,
    #     image_names, 
    # )

    # save_ply_visualization(pred, args.save_path)
    
    # 1. 准备起始点（在第一帧图像中选择要追踪的点）
    num_tracks = 100
    H, W = 518, 518  # DINO 图像尺寸

    # 假设有一个 mask，从中采样起始点
    mask = torch.ones(H, W)  # 或者是实际的分割 mask
    masked_coords = mask.squeeze().nonzero(as_tuple=False)
    indices = torch.randperm(len(masked_coords))[:num_tracks]
    start_points = masked_coords[indices][:, [1, 0]].float()  # (num_tracks, 2), (x, y) 格式

    tracking_result = model.track_points(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        dino_image_transform=dino_transform,
        images=images,
        start_points=start_points,
        source_view_idx=0,
    )
    # 3. 获取结果
    pred_tracks = tracking_result['pred_tracks']  # (num_tracks, V, 2)
    flow_maps = tracking_result['flow_maps']      # (V-1, 2, H, W)

    print(f"Tracked {num_tracks} points across {pred_tracks.shape[1]} views")
    
if __name__ == "__main__":
    main()