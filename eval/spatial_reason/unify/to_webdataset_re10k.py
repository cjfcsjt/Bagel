#!/usr/bin/env python3
"""
Convert RealEstate10K (RE10K) data into **WebDataset** shards and
(optionally) upload them to a HuggingFace Hub dataset repository.

Features
--------
* Reads the official RealEstate10K annotation layout at
  ``{annotation-root}/{train,test}/{scene_id}.txt``.
* Reads the corresponding video frame images from
  ``{image-root}/{scene_id}/{timestamp}.png``.
* Each WebDataset sample corresponds to **one scene** and contains:
  - All frame images (``{timestamp}.png``)
  - The annotation txt file (``annotation.txt``)
  - A ``meta.json`` with scene metadata (scene_id, split, n_frames, etc.)
* Supports **incremental mode**: re-running with the same ``--output-dir``
  will only process new / previously-skipped scenes.
* Supports optional ``--max-frames`` to subsample frames per scene.
* Shards are written as ``.tar`` files with configurable max samples per shard.

Usage
-----
python to_webdataset_re10k.py \\
    --annotation-root /path/to/RealEstate10K \\
    --image-root /path/to/RE10K_images \\
    --output-dir /path/to/wds_output \\
    --max-samples-per-shard 50 \\
    --num-workers 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  RE10K 标注文件解析
# ═══════════════════════════════════════════════════════════════════════

def parse_re10k_annotation(txt_path: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    解析 RealEstate10K 标注 txt 文件。

    格式：
      第一行: YouTube URL
      后续每行: timestamp fx fy cx cy k1 k2 R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz

    Returns
    -------
    (youtube_url, frames_list)
    """
    if not os.path.isfile(txt_path):
        return None, []

    with open(txt_path, "r") as f:
        lines = f.read().strip().splitlines()

    if not lines:
        return None, []

    youtube_url = lines[0].strip()
    frames = []

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 19:
            continue
        timestamp = parts[0]
        # fx, fy, cx, cy, k1, k2 = parts[1:7]
        # R(3x3) + t(3x1) 交错排列: R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz
        frames.append({
            "timestamp": timestamp,
            "intrinsics": [float(x) for x in parts[1:7]],
            "extrinsics": [float(x) for x in parts[7:19]],
        })

    return youtube_url, frames


# ═══════════════════════════════════════════════════════════════════════
#  场景数据加载
# ═══════════════════════════════════════════════════════════════════════

def process_single_scene(
    scene_id: str,
    split: str,
    annotation_path: str,
    image_dir: str,
    max_frames: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    处理单个 RE10K 场景。

    读取标注文件和对应的帧图片，返回一个 dict 用于写入 WebDataset tar shard。

    Parameters
    ----------
    scene_id : str
        场景 ID（如 "0000cc6d8b108390"）
    split : str
        "train" 或 "test"
    annotation_path : str
        标注 txt 文件路径
    image_dir : str
        该场景的图片目录路径（包含 {timestamp}.png）
    max_frames : int
        最大帧数（0 = 全部）

    Returns
    -------
    (sample_dict, None) 成功时
    (None, reason_string) 跳过时
    """
    # 解析标注文件
    youtube_url, frames = parse_re10k_annotation(annotation_path)
    if not frames:
        return None, f"标注文件为空或解析失败: {annotation_path}"

    # 检查图片目录
    if not os.path.isdir(image_dir):
        return None, f"图片目录不存在: {image_dir}"

    # 获取可用的图片文件
    available_images = set(os.listdir(image_dir))
    frame_timestamps = [f["timestamp"] for f in frames]

    # 匹配标注中的 timestamp 与实际图片
    matched_frames = []
    for frame in frames:
        ts = frame["timestamp"]
        img_name = f"{ts}.png"
        if img_name in available_images:
            matched_frames.append(frame)

    if not matched_frames:
        return None, f"没有找到匹配的图片（标注有 {len(frames)} 帧，图片目录有 {len(available_images)} 个文件）"

    # 可选：子采样帧
    if max_frames > 0 and len(matched_frames) > max_frames:
        step = len(matched_frames) / max_frames
        matched_frames = [matched_frames[int(i * step)] for i in range(max_frames)]

    # 构建 sample
    sample: Dict[str, Any] = {"__key__": scene_id}

    # 添加标注文件
    with open(annotation_path, "rb") as f:
        sample["annotation.txt"] = f.read()

    # 添加帧图片
    n_loaded = 0
    for frame in matched_frames:
        ts = frame["timestamp"]
        img_name = f"{ts}.png"
        img_path = os.path.join(image_dir, img_name)
        try:
            with open(img_path, "rb") as f:
                sample[f"frames/{img_name}"] = f.read()
            n_loaded += 1
        except Exception as e:
            # 跳过读取失败的图片
            continue

    if n_loaded == 0:
        return None, "所有图片读取失败"

    # 构建元数据
    meta = {
        "scene_id": scene_id,
        "split": split,
        "youtube_url": youtube_url,
        "n_frames_annotation": len(frames),
        "n_frames_available": len(available_images),
        "n_frames_loaded": n_loaded,
        "timestamps": [f["timestamp"] for f in matched_frames[:n_loaded]],
    }
    sample["meta.json"] = json.dumps(meta).encode("utf-8")

    return sample, None


# ═══════════════════════════════════════════════════════════════════════
#  WebDataset shard writer（支持增量模式）
# ═══════════════════════════════════════════════════════════════════════

class ShardWriter:
    """
    将 samples 写入 WebDataset 风格的 tar shards。

    每个 shard 命名为 ``{prefix}-{shard_idx:06d}.tar``，
    最多包含 ``max_samples`` 个 samples。

    支持 **增量模式**：如果 ``metadata.json`` 已存在于 *output_dir*，
    则保留已有 shards 并追加新的。
    """

    def __init__(self, output_dir: str, prefix: str = "re10k",
                 max_samples: int = 50):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_samples = max_samples

        os.makedirs(output_dir, exist_ok=True)

        # ── 尝试从已有 metadata 恢复 ──
        self._existing_shard_paths: List[str] = []
        self._existing_shard_scene_map: Dict[str, List[str]] = {}
        self._existing_scene_shard_map: Dict[str, str] = {}
        self._existing_total_samples: int = 0

        metadata_path = os.path.join(output_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                prev_meta = json.load(f)
            for shard_info in prev_meta.get("shards", []):
                shard_name = shard_info["shard_name"]
                shard_path = os.path.join(output_dir, shard_name)
                if os.path.isfile(shard_path):
                    self._existing_shard_paths.append(shard_path)
                    self._existing_shard_scene_map[shard_name] = shard_info.get("scenes", [])
            self._existing_scene_shard_map = dict(prev_meta.get("scene_to_shard", {}))
            self._existing_total_samples = prev_meta.get("total_scenes", 0)
            next_idx = len(self._existing_shard_paths)
            print(f"[增量模式] 发现已有 metadata，包含 "
                  f"{self._existing_total_samples} 个场景，"
                  f"{len(self._existing_shard_paths)} 个 shard。"
                  f"新 shard 从索引 {next_idx} 开始。")
        else:
            next_idx = 0

        self.shard_idx = next_idx
        self.sample_count = 0
        self.total_samples = 0  # 仅计数本次新增的 samples
        self.tar: Optional[tarfile.TarFile] = None
        self.shard_paths: List[str] = []  # 仅新 shard 路径
        self._current_shard_scenes: List[str] = []
        self._shard_scene_map: Dict[str, List[str]] = {}
        self._scene_shard_map: Dict[str, str] = {}
        self._new_shard_opened = False

    def get_existing_scene_ids(self) -> set:
        """返回已有 shards 中的场景 ID 集合。"""
        return set(self._existing_scene_shard_map.keys())

    def _open_new_shard(self):
        """打开一个新的 tar shard 文件。"""
        if self.tar is not None:
            self.tar.close()
            prev_shard = os.path.basename(self.shard_paths[-1])
            self._shard_scene_map[prev_shard] = list(self._current_shard_scenes)
            self._current_shard_scenes = []
            self.shard_idx += 1
        shard_name = f"{self.prefix}-{self.shard_idx:06d}.tar"
        shard_path = os.path.join(self.output_dir, shard_name)
        self.tar = tarfile.open(shard_path, "w")
        self.shard_paths.append(shard_path)
        self.sample_count = 0
        self._new_shard_opened = True

    def add_sample(self, sample: Dict[str, Any]):
        """添加一个 sample（场景）到当前 shard。"""
        if not self._new_shard_opened:
            self._open_new_shard()
        elif self.sample_count >= self.max_samples:
            self._open_new_shard()

        key = sample.pop("__key__")
        self._current_shard_scenes.append(key)
        self._scene_shard_map[key] = os.path.basename(self.shard_paths[-1])

        for field_name, data in sample.items():
            if isinstance(data, str):
                data = data.encode("utf-8")
            if not isinstance(data, bytes):
                continue

            member_name = f"{key}/{field_name}"
            info = tarfile.TarInfo(name=member_name)
            info.size = len(data)
            self.tar.addfile(info, io.BytesIO(data))

        self.sample_count += 1
        self.total_samples += 1

    def close(self):
        """关闭当前 shard 并写入合并后的 metadata。"""
        if self.tar is not None:
            self.tar.close()
            self.tar = None
            if self.shard_paths:
                last_shard = os.path.basename(self.shard_paths[-1])
                self._shard_scene_map[last_shard] = list(self._current_shard_scenes)
                self._current_shard_scenes = []

        # ── 合并已有 + 新增 ──
        all_shard_paths = self._existing_shard_paths + self.shard_paths
        all_shard_scene_map = dict(self._existing_shard_scene_map)
        all_shard_scene_map.update(self._shard_scene_map)
        all_scene_shard_map = dict(self._existing_scene_shard_map)
        all_scene_shard_map.update(self._scene_shard_map)
        all_total = self._existing_total_samples + self.total_samples

        if all_total == 0:
            return

        # ── 写入 shards.txt ──
        shards_txt_path = os.path.join(self.output_dir, "shards.txt")
        with open(shards_txt_path, "w") as f:
            for shard_path in all_shard_paths:
                shard_name = os.path.basename(shard_path)
                scenes = all_shard_scene_map.get(shard_name, [])
                f.write(f"{shard_name}\t{','.join(scenes)}\n")

        # ── 计算每个 shard 的文件大小 ──
        shard_info_list = []
        for shard_path in all_shard_paths:
            shard_name = os.path.basename(shard_path)
            file_size = os.path.getsize(shard_path) if os.path.isfile(shard_path) else 0
            shard_info_list.append({
                "shard_name": shard_name,
                "num_scenes": len(all_shard_scene_map.get(shard_name, [])),
                "scenes": all_shard_scene_map.get(shard_name, []),
                "file_size_bytes": file_size,
            })

        # ── 写入 metadata.json ──
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_scenes": all_total,
            "total_shards": len(all_shard_paths),
            "max_samples_per_shard": self.max_samples,
            "shards": shard_info_list,
            "scene_to_shard": all_scene_shard_map,
        }
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  写入 {shards_txt_path}")
        print(f"  写入 {metadata_path}")

    def summary(self) -> str:
        all_total = self._existing_total_samples + self.total_samples
        all_shards = len(self._existing_shard_paths) + len(self.shard_paths)
        parts = [
            f"总计: {all_total} 个场景，{all_shards} 个 shard，"
            f"输出目录: {self.output_dir}"
        ]
        if self._existing_total_samples > 0:
            parts.append(
                f"  (已有: {self._existing_total_samples} 个场景 / "
                f"{len(self._existing_shard_paths)} 个 shard，"
                f"新增: {self.total_samples} 个场景 / "
                f"{len(self.shard_paths)} 个 shard)"
            )
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  HuggingFace 上传
# ═══════════════════════════════════════════════════════════════════════

def upload_to_hf(output_dir: str, repo_id: str, private: bool = False):
    """将 output_dir 中的所有文件上传到 HuggingFace Hub 数据集仓库。"""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"[HF] 仓库 '{repo_id}' 已就绪。")
    except Exception as e:
        print(f"[HF] 创建仓库警告: {e}")

    files_to_upload = []
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            files_to_upload.append((fpath, fname))

    if not files_to_upload:
        print("[HF] 没有找到需要上传的文件。")
        return

    print(f"[HF] 正在上传 {len(files_to_upload)} 个文件 ...")

    for fpath, fname in tqdm(files_to_upload, desc="上传中"):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )

    readme_content = """---
configs:
  - config_name: default
    data_files:
      - split: train
        path: "re10k-*.tar"
---

# RealEstate10K WebDataset

This dataset contains **RealEstate10K** data converted to
[WebDataset](https://github.com/webdataset/webdataset) format.

All scenes are stored in flat `.tar` shards. Split files
(`re10k_train.txt`, `re10k_test.txt`) are included alongside the shards.

Each sample is one scene with:
- ``frames/{timestamp}.png``  — video frame images
- ``annotation.txt``          — original RE10K annotation (camera intrinsics & extrinsics)
- ``meta.json``               — scene metadata (scene_id, split, n_frames, youtube_url, etc.)

## Annotation Format

Each annotation `.txt` file has:
- Line 1: YouTube video URL
- Lines 2+: `timestamp fx fy cx cy k1 k2 R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz`
  - `fx, fy, cx, cy`: normalized camera intrinsics
  - `k1, k2`: radial distortion coefficients
  - `R(3x3), t(3x1)`: camera extrinsics (world-to-camera)
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] 上传完成 ({len(files_to_upload)} 个文件): "
          f"https://huggingface.co/datasets/{repo_id}")


# ═══════════════════════════════════════════════════════════════════════
#  多进程 Worker
# ═══════════════════════════════════════════════════════════════════════

def _worker(args_tuple):
    scene_id, split, annotation_path, image_dir, max_frames = args_tuple
    try:
        return process_single_scene(scene_id, split, annotation_path, image_dir, max_frames)
    except Exception as e:
        print(f"[WARN] 处理场景 {scene_id} 失败: {e}")
        return None, f"exception: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="将 RealEstate10K 转换为 WebDataset shards，"
                    "并可选上传到 HuggingFace Hub。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotation-root", type=str,
        default="/apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/realestate10k/RealEstate10K",
        help="RE10K 标注根目录（包含 train/ 和 test/ 子目录，内有 {scene_id}.txt）",
    )
    parser.add_argument(
        "--image-root", type=str,
        default="/apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/datasets--mutou0308--RE10K/snapshots/85f1c43d30031e1cf9764eb30f40daa3ec72f6ae",
        help="RE10K 图片根目录（包含 {scene_id}/ 子目录，内有 {timestamp}.png）",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出 WebDataset .tar shards 的目录",
    )
    parser.add_argument(
        "--max-samples-per-shard", type=int, default=50,
        help="每个 .tar shard 最大场景数",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="每个场景最大帧数（0 = 全部）。用于减小 shard 大小。",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "test"],
        help="要处理的数据集 split（如 train test）",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace 数据集仓库 ID（用于上传）",
    )
    parser.add_argument(
        "--hf-private", action="store_true", default=False,
        help="将 HuggingFace 仓库设为私有",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="并行 worker 数量（0 = 顺序处理）。"
             "注意：并行处理会将整个场景加载到内存中。",
    )
    args = parser.parse_args()

    annotation_root = args.annotation_root
    image_root = args.image_root

    if not os.path.isdir(annotation_root):
        print(f"[ERROR] 标注根目录不存在: {annotation_root}")
        sys.exit(1)
    if not os.path.isdir(image_root):
        print(f"[ERROR] 图片根目录不存在: {image_root}")
        sys.exit(1)

    # ── 发现所有场景 ──
    scene_entries: List[Tuple[str, str, str, str]] = []  # (scene_id, split, annotation_path, image_dir)
    split_scene_counts: Dict[str, int] = {}

    for split in args.splits:
        split_dir = os.path.join(annotation_root, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split 目录不存在: {split_dir}")
            continue

        txt_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".txt"))
        count = 0
        for txt_file in txt_files:
            scene_id = txt_file[:-4]  # 去掉 .txt 后缀
            annotation_path = os.path.join(split_dir, txt_file)
            image_dir = os.path.join(image_root, scene_id)
            # 只添加有对应图片目录的场景
            if os.path.isdir(image_dir):
                scene_entries.append((scene_id, split, annotation_path, image_dir))
                count += 1
        split_scene_counts[split] = count
        print(f"  发现 {split} 集: {len(txt_files)} 个标注文件，"
              f"{count} 个有对应图片目录")

    if not scene_entries:
        print(f"[ERROR] 没有找到任何有效场景")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 写入 split 文件到输出目录 ──
    for split in args.splits:
        split_scenes = [sid for sid, sp, _, _ in scene_entries if sp == split]
        if split_scenes:
            split_path = os.path.join(args.output_dir, f"re10k_{split}.txt")
            with open(split_path, "w") as f:
                f.write("\n".join(split_scenes) + "\n")
            print(f"  写入 split 文件: re10k_{split}.txt ({len(split_scenes)} 个场景)")

    total_scenes = len(scene_entries)
    print(f"\n[RE10K → WebDataset]")
    print(f"  标注根目录     : {annotation_root}")
    print(f"  图片根目录     : {image_root}")
    print(f"  输出目录       : {args.output_dir}")
    print(f"  总场景数       : {total_scenes} "
          f"({', '.join(f'{sp}: {c}' for sp, c in split_scene_counts.items())})")
    print(f"  每 shard 场景数: {args.max_samples_per_shard}")
    print(f"  最大帧数/场景  : {args.max_frames or '全部'}")
    print(f"  Workers        : {args.num_workers or '顺序处理'}")
    print()

    writer = ShardWriter(
        args.output_dir,
        prefix="re10k",
        max_samples=args.max_samples_per_shard,
    )

    # ── 增量模式：过滤已处理的场景 ──
    existing_scenes = writer.get_existing_scene_ids()
    if existing_scenes:
        new_entries = [(sid, sp, ap, idir) for sid, sp, ap, idir in scene_entries
                       if sid not in existing_scenes]
        print(f"[增量模式] {len(existing_scenes)} 个场景已在 shards 中，"
              f"{len(new_entries)} 个新场景待处理 "
              f"（共 {len(scene_entries)} 个）。")
    else:
        new_entries = scene_entries

    processed = 0
    skipped = 0
    already_done = len(scene_entries) - len(new_entries)

    if not new_entries:
        print("[增量模式] 没有新场景需要处理。")
    elif args.num_workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        import itertools
        import concurrent.futures

        tasks = [
            (sid, sp, ap, idir, args.max_frames)
            for sid, sp, ap, idir in new_entries
        ]
        max_inflight = args.num_workers * 2
        task_iter = iter(tasks)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for t in itertools.islice(task_iter, max_inflight):
                fut = executor.submit(_worker, t)
                futures[fut] = t[0]  # scene_id

            pbar = tqdm(total=len(tasks), desc="处理场景")
            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    scene_id = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"[WARN] 场景 {scene_id} 异常: {e}")
                        result = (None, f"exception: {e}")

                    sample, reason = result
                    if sample is None:
                        skipped += 1
                        print(f"[SKIP] {scene_id}: {reason}")
                    else:
                        writer.add_sample(sample)
                        del sample
                        processed += 1

                    pbar.update(1)
                    pbar.set_postfix(ok=processed, skip=skipped)

                    next_task = next(task_iter, None)
                    if next_task is not None:
                        fut = executor.submit(_worker, next_task)
                        futures[fut] = next_task[0]
            pbar.close()
    else:
        # ── 顺序处理 ──
        for scene_id, split, annotation_path, image_dir in tqdm(new_entries, desc="处理场景"):
            sample, reason = process_single_scene(
                scene_id, split, annotation_path, image_dir,
                max_frames=args.max_frames,
            )
            if sample is None:
                skipped += 1
                print(f"[SKIP] {scene_id}: {reason}")
                continue
            writer.add_sample(sample)
            del sample
            processed += 1

    writer.close()

    print(f"\n{'='*60}")
    print(f"[完成] {writer.summary()}")
    print(f"  本次运行 — 处理: {processed}, 跳过: {skipped}, "
          f"已存在: {already_done}")
    print(f"{'='*60}")

    # ── 上传到 HuggingFace ──
    if args.hf_repo:
        upload_to_hf(args.output_dir, args.hf_repo, private=args.hf_private)


if __name__ == "__main__":
    main()
