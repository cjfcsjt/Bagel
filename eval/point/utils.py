"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
# from hydra.utils import instantiate
import cv2

from .correspondence import project_3dto2d, torch_knn
from .transformations import transform_points_Rt

import albumentations as A_transforms
import numpy as np
import torch
import os
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as transform_F
from PIL import Image, ImageOps
from torch.linalg import cross
from torch.nn.functional import normalize
# from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def unproject_2d_to_3d(uv_points, depth_map, intrinsics):
    """
    Unproject 2D points to 3D camera coordinates.
    """
    N = uv_points.shape[0]
    if uv_points.numel() == 0:
        return torch.empty((0, 3), device=uv_points.device)

    K_inv = torch.inverse(intrinsics)
    uv_h = F.pad(uv_points, (0, 1), "constant", 1.0)

    H, W = depth_map.shape
    norm_u = 2.0 * uv_points[:, 0] / (W - 1) - 1.0
    norm_v = 2.0 * uv_points[:, 1] / (H - 1) - 1.0
    grid = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)

    sampled_depths = F.grid_sample(
        depth_map.unsqueeze(0).unsqueeze(0), grid, align_corners=True
    )
    depth_values = sampled_depths.view(N)

    xyz_points = (K_inv @ uv_h.T).T * depth_values[:, None]
    return xyz_points


def generate_dataset_report(
    all_dataset_2d_errors, all_dataset_3d_errors, model_name, log_dir
):
    """
    Aggregate dataset errors, print a report, and append a CSV line to a log.
    """
    log_filename = os.path.join(log_dir, f"tracking_report_{model_name}.log")
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not all_dataset_2d_errors:
        report_str = (
            "No visible GT points found in the entire dataset to calculate errors."
        )
        print(report_str)
        with open(log_filename, "a") as f:
            f.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(report_str + "\n")
        return

    final_2d_errors = torch.cat(all_dataset_2d_errors)
    final_3d_errors = torch.cat(all_dataset_3d_errors)

    ate_2d = final_2d_errors.mean().item()
    px_thresholds = [1, 2, 5, 10, 25, 50]
    accuracies_2d = {
        th: (final_2d_errors < th).float().mean().item() * 100 for th in px_thresholds
    }

    ate_3d = final_3d_errors.mean().item()
    m_thresholds = [0.01, 0.02, 0.05, 0.1]
    accuracies_3d = {
        th: (final_3d_errors < th).float().mean().item() * 100 for th in m_thresholds
    }

    report_lines = [
        "\n===================================================",
        f"--- Overall Dataset Tracking Performance Report for: {model_name} ---",
        f"--- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---",
        f"Evaluated on a total of {final_2d_errors.numel()} visible points.",
        "\n[2D Image Space Errors]",
        f"  - Average Trajectory Error (ATE): {ate_2d:.2f} pixels",
    ]
    for th, acc in accuracies_2d.items():
        report_lines.append(f"  - Accuracy @ {th}px: {acc:.2f}%")

    report_lines.append("\n[3D Metric Space Errors]")
    report_lines.append(f"  - Average Trajectory Error (ATE): {ate_3d * 100:.2f} cm")
    for th, acc in accuracies_3d.items():
        report_lines.append(f"  - Accuracy @ {th * 100:.0f}cm: {acc:.2f}%")

    csv_header = "model_name,total_points,ate_2d_px,acc_1px,acc_2px,acc_5px,acc_10px,acc_25px,acc_50px,ate_3d_cm,acc_1cm,acc_2cm,acc_5cm,acc_10cm"
    csv_values = [
        model_name,
        f"{final_2d_errors.numel()}",
        f"{ate_2d:.4f}",
        f"{accuracies_2d[1]:.4f}",
        f"{accuracies_2d[2]:.4f}",
        f"{accuracies_2d[5]:.4f}",
        f"{accuracies_2d[10]:.4f}",
        f"{accuracies_2d[25]:.4f}",
        f"{accuracies_2d[50]:.4f}",
        f"{ate_3d * 100:.4f}",
        f"{accuracies_3d[0.01]:.4f}",
        f"{accuracies_3d[0.02]:.4f}",
        f"{accuracies_3d[0.05]:.4f}",
        f"{accuracies_3d[0.10]:.4f}",
    ]
    csv_line = ",".join(csv_values)

    report_lines.append("\n--- CSV formatted line (for spreadsheets) ---")
    report_lines.append(csv_header)
    report_lines.append(csv_line)
    report_lines.append("===================================================\n")

    final_report = "\n".join(report_lines)
    print(final_report)

    with open(log_filename, "a") as f:
        f.write(final_report)
    print(f"Report appended to {log_filename}")


def compute_errors_for_sample(pred_tracks, gt_tracks, depths, intrinsics):
    """
    Compute 2D/3D tracking errors for a single sample.
    """
    _, num_views, _ = gt_tracks.shape

    all_2d_errors = []
    all_3d_errors = []

    for v_idx in range(num_views):
        visibility_mask = gt_tracks[:, v_idx, 2] > 0
        if not visibility_mask.any():
            continue

        visible_gt_uv = gt_tracks[visibility_mask, v_idx, :2]
        visible_pred_uv = pred_tracks[visibility_mask, v_idx, :]

        errors_2d_view = torch.linalg.norm(visible_pred_uv - visible_gt_uv, dim=1)
        all_2d_errors.append(errors_2d_view)

        depth_map_v = depths[v_idx, 0]
        intrinsics_v = intrinsics[v_idx]
        gt_xyz = unproject_2d_to_3d(visible_gt_uv, depth_map_v, intrinsics_v)
        pred_xyz = unproject_2d_to_3d(visible_pred_uv, depth_map_v, intrinsics_v)

        errors_3d_view = torch.linalg.norm(pred_xyz - gt_xyz, dim=1)
        all_3d_errors.append(errors_3d_view)

    if not all_2d_errors:
        return torch.tensor([]), torch.tensor([])

    final_2d_errors = torch.cat(all_2d_errors)
    final_3d_errors = torch.cat(all_3d_errors)
    return final_2d_errors, final_3d_errors


def is_visible(u_proj, v_proj, depth_proj, target_depth_map, W, H):
    """
    Check if a projected point is visible in the target view.
    """
    if u_proj is None or v_proj is None or depth_proj is None:
        return False

    if not (depth_proj > 0 and 0 <= u_proj < W and 0 <= v_proj < H):
        return False

    u_int, v_int = int(round(u_proj)), int(round(v_proj))
    if not (0 <= u_int < W and 0 <= v_int < H):
        return False

    depth_from_map = target_depth_map[v_int, u_int].item()
    epsilon = 0.05
    if depth_from_map > 0 and depth_proj > (depth_from_map + epsilon):
        return False

    return True


def project_point(
    u,
    v,
    depth_source,
    intrinsics_source,
    Rt_source,
    intrinsics_target,
    Rt_target,
    source_inverse,
    target_inverse,
):
    """
    Project a point from the source image into the target image.
    """
    depth_value = depth_source[v, u].item()
    if depth_value <= 0:
        return None, None, None

    u_center, v_center = u + 0.5, v + 0.5
    uv_one = torch.tensor(
        [u_center, v_center, 1.0], dtype=torch.float32, device=depth_source.device
    )
    K_inv = torch.inverse(intrinsics_source)

    point_cam_source = (K_inv @ uv_one) * depth_value
    point_cam_source = point_cam_source.unsqueeze(0)

    point_world = transform_points_Rt(
        point_cam_source, Rt_source, inverse=source_inverse
    )
    point_cam_target = transform_points_Rt(
        point_world, Rt_target, inverse=target_inverse
    )

    depth_in_target = point_cam_target[0, 2].item()
    uv_target = project_3dto2d(point_cam_target, intrinsics_target)

    final_uv = uv_target.squeeze()
    u_final = final_uv[0].item() - 0.5
    v_final = final_uv[1].item() - 0.5
    return u_final, v_final, depth_in_target


def calculate_gt_tracks(
    start_points,
    depths,
    intrinsics,
    Rts,
    source_inverse,
    target_inverse,
    record_all_projections,
    check_start_in_bounds,
    invalidate_if_never_visible,
):
    """
    Compute GT tracks with configurable visibility and projection handling.
    """
    num_tracks, _ = start_points.shape
    V, _, H, W = depths.shape
    gt_tracks = torch.full((num_tracks, V, 3), -1.0, device=start_points.device)
    gt_tracks[:, 0, 0:2] = start_points
    gt_tracks[:, 0, 2] = 1.0

    source_depth = depths[0, 0]
    source_intrinsics = intrinsics[0]
    source_Rt = Rts[0]

    for i in range(num_tracks):
        start_u, start_v = int(start_points[i, 0]), int(start_points[i, 1])

        if check_start_in_bounds and not (0 <= start_u < W and 0 <= start_v < H):
            gt_tracks[i, :, 2] = -1.0
            continue

        for v_idx in range(1, V):
            target_intrinsics = intrinsics[v_idx]
            target_Rt = Rts[v_idx]
            target_depth_map = depths[v_idx, 0]

            u_proj, v_proj, depth_proj = project_point(
                start_u,
                start_v,
                source_depth,
                source_intrinsics,
                source_Rt,
                target_intrinsics,
                target_Rt,
                source_inverse,
                target_inverse,
            )

            if record_all_projections and u_proj is not None:
                gt_tracks[i, v_idx, 0] = u_proj
                gt_tracks[i, v_idx, 1] = v_proj

            if is_visible(u_proj, v_proj, depth_proj, target_depth_map, W, H):
                gt_tracks[i, v_idx, 0] = u_proj
                gt_tracks[i, v_idx, 1] = v_proj
                gt_tracks[i, v_idx, 2] = 1.0

        if invalidate_if_never_visible:
            is_visible_elsewhere = torch.any(gt_tracks[i, 1:, 2] > 0)
            if not is_visible_elsewhere:
                gt_tracks[i, 0, 2] = -1.0

    return gt_tracks


def find_correspondences_feature_based(uv_s, feat_s, feat_t):
    """
    Find correspondences in the target feature map for source points.
    """
    _, C, Hf, Wf = feat_s.shape

    N = uv_s.shape[0]
    norm_u = 2.0 * uv_s[:, 0] / (Wf - 1) - 1.0
    norm_v = 2.0 * uv_s[:, 1] / (Hf - 1) - 1.0
    grid_query = torch.stack([norm_u, norm_v], dim=1).view(1, 1, -1, 2)
    sampled_feats = F.grid_sample(feat_s, grid_query, align_corners=False)
    query_feats = sampled_feats.permute(0, 3, 2, 1).reshape(N, C)

    target_feats = feat_t.view(C, -1).T
    _, nn_indices = torch_knn(query_feats, target_feats, k=1)

    matched_indices = nn_indices.squeeze(-1)
    matched_v = matched_indices // Wf
    matched_u = matched_indices % Wf
    uv_t = torch.stack([matched_u, matched_v], dim=1).float()
    return uv_t


def show_batch(
    batch_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], nrow=4, step=0
):
    """
    Save a batch of images as a grid.
    """
    images = batch_tensor.clone().detach().cpu()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    images = images * std + mean

    grid_img = torchvision.utils.make_grid(
        images, nrow=nrow, padding=2, normalize=False
    )
    torchvision.utils.save_image(grid_img, "{}.png".format(step))


def softmax_with_temperature(x, beta, d=1):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M
    exp_x = torch.exp(x / beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum


def soft_argmax(corr, x_normal, y_normal, beta=0.02):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    b, _, h, w = corr.size()
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1, h, w, h, w)

    grid_x = corr.sum(dim=1, keepdim=False)
    x_normal = x_normal.expand(b, w)
    x_normal = x_normal.view(b, w, 1, 1)
    grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)

    grid_y = corr.sum(dim=2, keepdim=False)
    y_normal = y_normal.expand(b, h)
    y_normal = y_normal.view(b, h, 1, 1)
    grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)
    return grid_x, grid_y


def unnormalise_and_convert_mapping_to_flow(map):
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    mapping[:, 0, :, :] = (
        (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0
    )
    mapping[:, 1, :, :] = (
        (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0
    )

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


def get_backward_flow_from_attn(
    source_idx, target_idx, attn_maps, hp, wp, patch_size, H, W
):
    """
    Compute dense flow from attention maps.
    Note: returns flow from target to source.
    """
    device = attn_maps[0].device
    B = attn_maps[0].shape[0]

    camap_s_to_t = [attn[:, source_idx, :, target_idx] for attn in attn_maps]
    camap_s_to_t = torch.stack(camap_s_to_t, dim=1).mean(dim=1)

    camap_t_to_s = [attn[:, target_idx, :, source_idx] for attn in attn_maps]
    camap_t_to_s = torch.stack(camap_t_to_s, dim=1).mean(dim=1)

    refined_corr = (camap_s_to_t + camap_t_to_s.transpose(-1, -2)) / 2

    x_normal = torch.linspace(-1, 1, wp, device=device)
    y_normal = torch.linspace(-1, 1, hp, device=device)

    grid_x, grid_y = soft_argmax(
        refined_corr.view(B, -1, hp, wp), x_normal, y_normal, beta=0.0001
    )

    coarse_flow = torch.cat((grid_x, grid_y), dim=1)
    flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow)
    flow_est = F.interpolate(
        flow_est, size=(H, W), mode="bilinear", align_corners=False
    )
    flow_est[:, 0, :, :] *= W / wp
    flow_est[:, 1, :, :] *= H / hp

    return flow_est


def set_all_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_boustrophedon_grid_pos(v_idx, u_coord, v_coord, H, W, nrow, padding):
    grid_row = v_idx // nrow
    v_col_in_row = v_idx % nrow

    if grid_row % 2 == 1:
        grid_col = (nrow - 1) - v_col_in_row
    else:
        grid_col = v_col_in_row

    grid_u = padding + u_coord + grid_col * (W + padding)
    grid_v = padding + v_coord + grid_row * (H + padding)

    return int(grid_u), int(grid_v)


def visualize_comparison_tracks(images, pred_tracks, gt_tracks, save_path, nrow=4):
    # mean, std = (
    #     torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
    #     torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
    # )
    images_unnorm = [(img.cpu()).clamp(0, 1) for img in images]

    num_views = len(images_unnorm)
    images_reordered = []
    
    # 行0（正序）:  [0]  [1]  [2]  [3]
    # 行1（翻转）:  [7]  [6]  [5]  [4]
    # 行2（正序）:  [8]  [9]  [10] [11]
    for i in range(0, num_views, nrow): # 每次取 nrow 张图像作为"一行"，决定了每行放几张图。
        chunk = images_unnorm[i : i + nrow]
        if (i // nrow) % 2 == 1:
            images_reordered.extend(chunk[::-1]) # 奇数行翻转
        else:
            images_reordered.extend(chunk) # 偶数行正序

    padding = 4
    grid_img_tensor = torchvision.utils.make_grid(
        images_reordered, nrow=nrow, padding=padding
    )

    base_img = cv2.cvtColor(
        (grid_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    # 分别为 pred 和 gt 复制一份底图
    pred_img = base_img.copy()
    gt_img = base_img.copy()
    vis_img = base_img.copy()

    _, H, W = images[0].shape
    pred_color, gt_color, invisible_color = (255, 0, 0), (0, 255, 0), (0, 0, 255)  # BGR: 蓝色, 绿色, 红色

    for i in range(pred_tracks.shape[0]):
        # --- 绘制预测轨迹（蓝色）到 pred_img 和 vis_img ---
        for v in range(num_views):
            gt_pt_data = gt_tracks[i, v].cpu().numpy()
            is_visible_flag = gt_pt_data[2] > 0
            if not is_visible_flag:
                continue

            pred_pt = pred_tracks[i, v].cpu().numpy()
            u, v_coord = pred_pt[0], pred_pt[1]
            print(f"pred_pt: u={u}, v={v_coord}")  # 检查预测值是否为负
            if u < 0 or v_coord < 0:
                continue

            grid_u, grid_v = _get_boustrophedon_grid_pos( # 当前单张图像内的像素坐标 (u, v_coord) 转换为网格拼图中的全局像素坐标 (grid_u, grid_v)
                v, u, v_coord, H, W, nrow, padding
            )
            cv2.circle(pred_img, (grid_u, grid_v), 5, pred_color, -1)
            cv2.circle(vis_img, (grid_u, grid_v), 5, pred_color, -1)

            if v > 0: # 在相邻两个视角的预测点之间画一条蓝色连线，用来可视化预测轨迹的走向。
                prev_gt_pt_data = gt_tracks[i, v - 1].cpu().numpy()
                if prev_gt_pt_data[2] > 0: # 用 GT 可见性作为条件：只有 GT 认为上一帧可见，才考虑画连线
                    prev_pred_pt = pred_tracks[i, v - 1].cpu().numpy()
                    if np.all(prev_pred_pt >= 0):
                        prev_u, prev_v_coord = prev_pred_pt[0], prev_pred_pt[1] # 取上一视角的预测点坐标，并检查坐标是否有效（≥ 0，排除无效的负值预测）
                        prev_grid_u, prev_grid_v = _get_boustrophedon_grid_pos( # 将上一视角预测点的局部像素坐标 → 网格图全局坐标
                            v - 1, prev_u, prev_v_coord, H, W, nrow, padding
                        )
                        cv2.line(pred_img, (prev_grid_u, prev_grid_v), (grid_u, grid_v), pred_color, 2)
                        cv2.line(vis_img, (prev_grid_u, prev_grid_v), (grid_u, grid_v), pred_color, 2)

        # --- 绘制 GT 轨迹（绿色）到 gt_img 和 vis_img ---
        for v in range(num_views):
            gt_pt_data = gt_tracks[i, v].cpu().numpy()
            is_visible_flag = gt_pt_data[2] > 0
            if not is_visible_flag:
                continue

            u, v_coord = gt_pt_data[0], gt_pt_data[1]

            grid_u, grid_v = _get_boustrophedon_grid_pos(
                v, u, v_coord, H, W, nrow, padding
            )
            center_point = (grid_u, grid_v)

            cv2.circle(gt_img, center_point, 6, gt_color, 2)
            cv2.circle(vis_img, center_point, 6, gt_color, 2)

            # 始终连接到第0帧（anchor帧），而不是前一帧
            if v > 0:
                anchor_gt_pt_data = gt_tracks[i, 0].cpu().numpy()
                if anchor_gt_pt_data[2] > 0:
                    anchor_u, anchor_v_coord = anchor_gt_pt_data[0], anchor_gt_pt_data[1]
                    anchor_grid_u, anchor_grid_v = _get_boustrophedon_grid_pos(
                        0, anchor_u, anchor_v_coord, H, W, nrow, padding
                    )
                    cv2.line(gt_img, (anchor_grid_u, anchor_grid_v), center_point, gt_color, 2)
                    cv2.line(vis_img, (anchor_grid_u, anchor_grid_v), center_point, gt_color, 2)

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存三张图：仅预测、仅GT、对比图
    pred_save_path = save_path.replace(".png", "_pred.png").replace(".jpg", "_pred.jpg")
    gt_save_path   = save_path.replace(".png", "_gt.png").replace(".jpg", "_gt.jpg")
    cv2.imwrite(pred_save_path, pred_img)
    cv2.imwrite(gt_save_path, gt_img)
    cv2.imwrite(save_path, vis_img)
    print(f"Pred-only visualization saved to {pred_save_path}")
    print(f"GT-only   visualization saved to {gt_save_path}")
    print(f"Comparison visualization saved to {save_path}")


def visualize_gt_tracks_step_by_step(images, gt_tracks, save_dir, nrow=4):
    """
    每画一条 GT 连线就保存一张图，用于逐步调试轨迹连线是否正确。
    文件命名格式：step_{track_i:03d}_view_{v:03d}.png
    """
    images_unnorm = [(img.cpu()).clamp(0, 1) for img in images]
    num_views = len(images_unnorm)
    images_reordered = []
    for i in range(0, num_views, nrow):
        chunk = images_unnorm[i : i + nrow]
        if (i // nrow) % 2 == 1:
            images_reordered.extend(chunk[::-1])
        else:
            images_reordered.extend(chunk)

    padding = 4
    grid_img_tensor = torchvision.utils.make_grid(
        images_reordered, nrow=nrow, padding=padding
    )
    base_img = cv2.cvtColor(
        (grid_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    _, H, W = images[0].shape
    gt_color = (0, 255, 0)  # BGR 绿色

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    step = 0
    canvas = base_img.copy()

    for i in range(gt_tracks.shape[0]):
        for v in range(num_views):
            gt_pt_data = gt_tracks[i, v].cpu().numpy()
            if gt_pt_data[2] <= 0:
                continue

            u, v_coord = gt_pt_data[0], gt_pt_data[1]
            grid_u, grid_v = _get_boustrophedon_grid_pos(v, u, v_coord, H, W, nrow, padding)
            center_point = (grid_u, grid_v)

            cv2.circle(canvas, center_point, 6, gt_color, 2)

            # 始终连接到第0帧（anchor帧），而不是前一帧
            if v > 0:
                anchor_gt_pt_data = gt_tracks[i, 0].cpu().numpy()
                if anchor_gt_pt_data[2] > 0:
                    anchor_u, anchor_v_coord = anchor_gt_pt_data[0], anchor_gt_pt_data[1]
                    anchor_grid_u, anchor_grid_v = _get_boustrophedon_grid_pos(
                        0, anchor_u, anchor_v_coord, H, W, nrow, padding
                    )
                    cv2.line(canvas, (anchor_grid_u, anchor_grid_v), center_point, gt_color, 2)

                    # 每画一条线保存一张图
                    step_path = os.path.join(save_dir, f"step_{step:04d}_track{i:03d}_v00_to_v{v:02d}.png")
                    cv2.imwrite(step_path, canvas)
                    print(f"Saved: {step_path}")
                    step += 1