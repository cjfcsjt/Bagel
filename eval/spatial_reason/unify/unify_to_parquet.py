#!/usr/bin/env python3
"""
Unify multiple spatial-reasoning datasets into Parquet files compatible with
``ReconthenUndIterableDataset`` / ``ParquetStandardIterableDataset``.

Supported datasets
------------------
1. **SPAR**        – JSON produced by ``generate_data_josn.py`` (already unified)
2. **MindCube**    – JSONL benchmark (image-only, no depth/pose)
3. **OmniSpatial** – JSON benchmark  (image-only, no depth/pose)
4. **OST-Bench**   – JSON (LLaVA interleaved or plain eval format, image-only)
5. **SenseNova-SI** – JSONL (832K spatial intelligence samples, image-only)
6. **RE10K**       – RealEstate10K multi-view scene folders (image-only, no depth/pose)
7. **ARKitScenes** – Apple ARKit indoor scene scans (RGB + depth + poses + intrinsics)
8. **ScanQA**      – ScanNet 3D scene QA (with depth, poses, intrinsics from EmbodiedScan)
9. **SPLD**        – SpatialLadder-26k (spatial reasoning + grounding, image-only)
10. **VGLLM**      – VG-LLM-Data series (scan2cap, scanrefer, scannet_det, sqa3d,
                      spar_234k, spar_7m, llava_hound; each saved to separate parquet)


Target Parquet row schema (matches ``parse_row`` in ``interleave_t2i_dataset.py``):
    question        : string
    answer          : string
    scene_name      : string   ('scannet', 'matterport3d', '3rscan', 'scannetpp',
                                 'structured3d', 'mindcube', 'omnispatial', 'ost')
    dataset_name    : string   (must contain 'spar' for depth/pose branch)
                                 'sensenova_si'
    image_list      : list<string>          – absolute RGB image paths
    depth_list      : list<string>          – absolute depth map paths (empty for image-only)
    poses           : list<list<float64>>   – per-frame 4×4 extrinsics (16 floats each)
    intrinsic       : list<float64>         – 4×4 camera intrinsic   (16 floats)
    depth_intrinsic : list<float64>         – 4×4 depth intrinsic    (16 floats)
    metadata        : string                – JSON with at least 'type' and 'id'

Usage
-----
# All four datasets at once:
python eval/spatial_reason/unify/unify_to_parquet.py \\
    --spar-json /path/to/spar_7m.json \\
    --mindcube-json /path/to/MindCube_train_80k.json \\
    --mindcube-data-root /path/to/reason_data \\
    --omnispatial-path /path/to/OmniSpatial-test/ \\
    --omnispatial-prompt-type manual_cot \\
    --omnispatial-eval-type re \\
    --ost-file /path/to/OST_bench_train_interleave_llava.json \\
    --ost-image-root /path/to/img_train/ \\
    --output-dir /path/to/output_parquets \\
    --rows-per-row-group 500 \\
    --rows-per-file 5000

# Only SPAR:
python eval/spatial_reason/unify/unify_to_parquet.py \\
    --spar-json /path/to/spar_7m.json \\
    --output-dir /path/to/output_parquets
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

# ─── Ensure the unify package is importable ──────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from spar import SPARDataset, build_datasets as spar_build_datasets  # noqa: E402
from mindcube import MindCubeDataset           # noqa: E402
from omnispatial import OmniSpatialDataset     # noqa: E402
from ost import OSTBenchDataset                # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
#  Image statistics collector
# ═════════════════════════════════════════════════════════════════════════

class ImageStatsCollector:
    """Collect per-sample image count distribution and large-sample examples."""

    def __init__(self, name: str, large_threshold: int = 10, max_large_examples: int = 5):
        self.name = name
        self.large_threshold = large_threshold
        self.max_large_examples = max_large_examples
        self.count_distribution: Dict[int, int] = defaultdict(int)  # n_images -> count
        self.large_examples: List[dict] = []  # collected cases with > threshold images
        self.total_images = 0
        self.total_samples = 0

    def record(self, n_images: int, sample_info: Optional[dict] = None):
        """Record one sample's image count."""
        self.count_distribution[n_images] += 1
        self.total_images += n_images
        self.total_samples += 1
        if n_images > self.large_threshold and len(self.large_examples) < self.max_large_examples:
            self.large_examples.append({
                "n_images": n_images,
                "info": sample_info or {},
            })

    def print_summary(self):
        """Print the collected statistics."""
        print(f"\n{'='*60}")
        print(f"  [{self.name}] Image count distribution summary")
        print(f"{'='*60}")
        print(f"  Total samples : {self.total_samples}")
        print(f"  Total images  : {self.total_images}")
        if self.total_samples > 0:
            print(f"  Avg images/sample: {self.total_images / self.total_samples:.2f}")

        # Print distribution sorted by image count
        print(f"\n  {'n_images':>10}  {'count':>8}  {'percent':>8}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
        for n in sorted(self.count_distribution.keys()):
            cnt = self.count_distribution[n]
            pct = cnt / self.total_samples * 100 if self.total_samples else 0
            print(f"  {n:>10}  {cnt:>8}  {pct:>7.2f}%")

        # Print large-sample examples
        n_large = sum(v for k, v in self.count_distribution.items() if k > self.large_threshold)
        print(f"\n  Samples with > {self.large_threshold} images: {n_large}")
        if self.large_examples:
            print(f"  Example cases (up to {self.max_large_examples}):")
            for i, ex in enumerate(self.large_examples):
                print(f"    [{i+1}] n_images={ex['n_images']}")
                info = ex["info"]
                for k, v in info.items():
                    val_str = str(v)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"        {k}: {val_str}")
        print(f"{'='*60}\n")


# ═════════════════════════════════════════════════════════════════════════
#  Arrow schema
# ═════════════════════════════════════════════════════════════════════════

ARROW_SCHEMA = pa.schema([
    ("question",        pa.string()),
    ("answer",          pa.string()),
    ("scene_name",      pa.string()),
    ("dataset_name",    pa.string()),
    ("image_path",      pa.list_(pa.string())),
    # ("image_bytes",     pa.list_(pa.binary())),              # raw RGB .jpg bytes
    # ("depth_bytes",     pa.list_(pa.binary())),              # raw depth .png bytes
    ("poses",           pa.list_(pa.list_(pa.float64()))),   # N × 16
    ("intrinsic",       pa.list_(pa.float64())),             # 16
    ("depth_intrinsic", pa.list_(pa.float64())),             # 16
    ("metadata",        pa.string()),
])

# 4×4 identity as 16-element list – used as placeholder for image-only datasets
_IDENTITY_4x4 = np.eye(4).flatten().tolist()

# Video3DLLM schema: 在 ARROW_SCHEMA 基础上增加离线预计算的 3D 字段
# - depth_list:   list<string>  每帧深度图路径
# - world_coords: string        JSON 序列化的 (V, H, W, 3) 世界坐标，float32
# - boundary:     list<float64> [x_min, x_max, y_min, y_max, z_min, z_max]
# - objects:      string        JSON 序列化的物体框列表 [[x,y,z,...], ...]
VIDEO3DLLM_ARROW_SCHEMA = pa.schema([
    ("question",        pa.string()),
    ("answer",          pa.string()),
    ("scene_name",      pa.string()),
    ("dataset_name",    pa.string()),
    ("image_path",      pa.list_(pa.string())),
    ("depth_list",      pa.list_(pa.string())),
    ("poses",           pa.list_(pa.list_(pa.float64()))),   # N × 16
    ("intrinsic",       pa.list_(pa.float64())),             # 16
    ("depth_intrinsic", pa.list_(pa.float64())),             # 16
    ("metadata",        pa.string()),
    # ── 离线预计算的 3D 字段 ──
    ("world_coords",    pa.string()),                        # JSON: (V, H, W, 3) float32
    ("boundary",        pa.list_(pa.float64())),             # [x_min, x_max, y_min, y_max, z_min, z_max]
    ("objects",         pa.string()),                        # JSON: list of object boxes
])


# ═════════════════════════════════════════════════════════════════════════
#  Per-dataset row converters
# ═════════════════════════════════════════════════════════════════════════

def _deep_destringify(obj):
    """递归地把嵌套的字符串化 JSON 还原为原生 Python 对象。
    
    例如 {"spar_info": "{\"bbox\": [0.1]}"} 会被还原为
    {"spar_info": {"bbox": [0.1]}}，避免写入 Parquet 时产生双重序列化。
    """
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, (dict, list)):
                return _deep_destringify(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return obj
    if isinstance(obj, dict):
        return {k: _deep_destringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_deep_destringify(item) for item in obj)
    return obj


def _ensure_metadata_str(meta) -> str:
    """Normalise metadata to a clean JSON string (no double-serialisation)."""
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, ValueError):
            return meta
    if isinstance(meta, dict):
        meta = _deep_destringify(meta)
        return json.dumps(meta, ensure_ascii=False)
    return json.dumps({}, ensure_ascii=False)


def _spar_item_to_row(data_item: dict, image_root: str, ds_name: str, need_3d_annotation: bool = False) -> Optional[dict]:
    """
    Convert a single SPAR annotation line (JSON object) into the target
    Parquet row schema.

    Each SPAR annotation line has:
      - ``image``: str or list[str]  – relative image paths
      - ``conversations``: [{from, value}, ...]  – human/gpt turns
      - ``type``: str  – task type (e.g. 'obj_spatial_relation_oo')
      - ``id``: str/int
      - optional drawing metadata fields consumed by ``draw_marker``
    
    Args:
        data_item: The annotation data item
        image_root: Root path for images
        ds_name: Dataset name
        need_3d_annotation: Whether to load 3D annotation data (poses, depth, intrinsics)
    """
    # ── resolve image paths ──
    raw_images = data_item.get("image", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    def _resolve(p):
        if p.startswith("s3://"):
            return image_root + p
        return os.path.join(image_root, p)

    image_list = [_resolve(p) for p in raw_images]

    # ── extract question / answer from conversations ──
    convs = data_item.get("conversations", [])
    question = ""
    answer = ""
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            question = value
        elif role == "gpt":
            answer = value

    # ── metadata (the entire data_item serves as drawing metadata) ──
    meta = {
        "type": data_item.get("type", "unknown"),
        "id": data_item.get("id", ""),
    }
    # Copy any extra keys that draw_marker may need
    for k, v in data_item.items():
        if k not in ("image", "conversations", "type", "id"):
            meta[k] = v

    n_images = len(image_list)

    # ── 3D annotation handling ──
    if need_3d_annotation:
        # Load 3D annotation from corresponding folder
        depth_list = _load_depth_paths(data_item, image_root, raw_images)
        poses = _load_poses(data_item, image_root, n_images)
        intrinsic = _load_intrinsic(data_item, image_root)
        depth_intrinsic = _load_depth_intrinsic(data_item, image_root)
    else:
        # No 3D annotation needed - set all to None
        depth_list = None
        poses = None
        intrinsic = None
        depth_intrinsic = None

    if 'rxr' in ds_name:
        scene_name = 'matterport3d'
    elif 'scannetpp' in ds_name:
        scene_name = 'scannetpp'
    elif 'scannet' in ds_name:
        scene_name = 'scannet'
    elif 'structured3d' in ds_name:
        scene_name = 'structured3d'
    
    return {
        "question":        question,
        "answer":          answer,
        "scene_name":      scene_name,
        "dataset_name":    f"spar_{ds_name}",
        "image_path":      image_list,
        "depth_list":      depth_list,
        "poses":           poses,
        "intrinsic":       intrinsic,
        "depth_intrinsic": depth_intrinsic,
        "metadata":        _ensure_metadata_str(meta),
    }


def _load_depth_paths(data_item: dict, image_root: str, raw_images: list) -> Optional[list]:
    """Load depth paths from annotation or corresponding folder."""
    depth_paths = data_item.get("depth", [])
    if isinstance(depth_paths, str):
        depth_paths = [depth_paths]
    if depth_paths:
        return [os.path.join(image_root, p) for p in depth_paths]
    # Try to infer depth paths from image paths
    # Customize this logic based on your folder structure
    return []


def _load_poses(data_item: dict, image_root: str, n_images: int) -> Optional[list]:
    """Load camera poses from annotation or corresponding folder."""
    poses = data_item.get("poses")
    if poses is not None:
        return poses
    # Try to load from file if path is specified
    pose_file = data_item.get("pose_file")
    if pose_file:
        pose_path = os.path.join(image_root, pose_file)
        if os.path.exists(pose_path):
            # Load poses from file - adjust format as needed
            import numpy as np
            poses_data = np.load(pose_path)
            return poses_data.tolist()
    return [_IDENTITY_4x4] * n_images


def _load_intrinsic(data_item: dict, image_root: str) -> Optional[list]:
    """Load camera intrinsic matrix from annotation or corresponding folder."""
    intrinsic = data_item.get("intrinsic")
    if intrinsic is not None:
        return intrinsic
    intrinsic_file = data_item.get("intrinsic_file")
    if intrinsic_file:
        intrinsic_path = os.path.join(image_root, intrinsic_file)
        if os.path.exists(intrinsic_path):
            import numpy as np
            return np.load(intrinsic_path).tolist()
    return _IDENTITY_4x4


def _load_depth_intrinsic(data_item: dict, image_root: str) -> Optional[list]:
    """Load depth camera intrinsic matrix from annotation or corresponding folder."""
    depth_intrinsic = data_item.get("depth_intrinsic")
    if depth_intrinsic is not None:
        return depth_intrinsic
    depth_intrinsic_file = data_item.get("depth_intrinsic_file")
    if depth_intrinsic_file:
        depth_intrinsic_path = os.path.join(image_root, depth_intrinsic_file)
        if os.path.exists(depth_intrinsic_path):
            import numpy as np
            return np.load(depth_intrinsic_path).tolist()
    return _IDENTITY_4x4


def _imageonly_item_to_row(item: dict) -> Optional[dict]:
    """
    Convert an image-only item (MindCube / OmniSpatial / OST-Bench) to
    the target schema.  Depth, poses, and intrinsics are filled with
    placeholders (empty list / identity matrix).
    """
    image_list = item.get("image_list", []) or item.get("image_paths", [])
    if not image_list:
        return None

    n_images = len(image_list)
    meta = item.get("metadata", {})

    return {
        "question":        item.get("question", ""),
        "answer":          item.get("answer", ""),
        "scene_name":      item.get("scene_name", ""),
        "dataset_name":    item.get("dataset_name", ""),
        "image_path":      image_list,
        "depth_list":      [],               # no depth for image-only
        "poses":           [_IDENTITY_4x4] * n_images,  # placeholder poses
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Parquet writer helpers
# ═════════════════════════════════════════════════════════════════════════

def flush_to_parquet(
    rows: List[dict],
    schema: pa.Schema,
    output_path: str,
    rows_per_row_group: int,
):
    """Write a list of row-dicts into a single Parquet file with multiple row groups."""
    if not rows:
        return

    columns = {field.name: [] for field in schema}
    for r in rows:
        for k in columns:
            columns[k].append(r[k])

    table = pa.table(columns, schema=schema)
    writer = pq.ParquetWriter(output_path, schema)

    total = len(rows)
    for start in range(0, total, rows_per_row_group):
        end = min(start + rows_per_row_group, total)
        writer.write_table(table.slice(start, end - start))

    writer.close()
    n_rg = (total + rows_per_row_group - 1) // rows_per_row_group
    print(f"  ✓ {total} rows → {output_path}  ({n_rg} row groups)")


class ParquetInfoTracker:
    """
    Shared tracker that incrementally maintains ``parquet_info.json``.

    Every time a new Parquet file is written, call ``register()`` to
    append its metadata and immediately persist to disk.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.info_path = os.path.join(output_dir, "parquet_info.json")
        # Load existing info if resuming
        if os.path.exists(self.info_path):
            with open(self.info_path, "r") as f:
                self.info: Dict[str, dict] = json.load(f)
            print(f"[ParquetInfo] Resumed with {len(self.info)} existing entries")
        else:
            self.info: Dict[str, dict] = {}

    def register(self, parquet_path: str):
        """Read metadata from a just-written Parquet file and persist."""
        try:
            pf = pq.ParquetFile(parquet_path)
            self.info[parquet_path] = {
                "num_row_groups": pf.metadata.num_row_groups,
                "num_rows": pf.metadata.num_rows,
            }
        except Exception as e:
            print(f"  [WARN] cannot read parquet metadata for {parquet_path}: {e}")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.info_path, "w") as f:
            json.dump(self.info, f, indent=2)

    def summary(self):
        total_files = len(self.info)
        total_rows = sum(v.get("num_rows", 0) for v in self.info.values())
        print(f"\n✓ parquet_info.json → {self.info_path}  "
              f"({total_files} files, {total_rows} total rows)")


class ParquetBufferedWriter:
    """Accumulates rows and flushes to numbered Parquet files."""

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        schema: pa.Schema,
        rows_per_file: int,
        rows_per_row_group: int,
        info_tracker: Optional[ParquetInfoTracker] = None,
    ):
        self.output_dir = output_dir
        self.prefix = prefix
        self.schema = schema
        self.rows_per_file = rows_per_file
        self.rows_per_row_group = rows_per_row_group
        self.info_tracker = info_tracker
        self.buffer: List[dict] = []
        self.file_counter = 0
        self.total_written = 0

    def add(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= self.rows_per_file:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_{self.file_counter:05d}.parquet",
        )
        flush_to_parquet(self.buffer, self.schema, path, self.rows_per_row_group)
        self.total_written += len(self.buffer)
        self.buffer = []
        self.file_counter += 1

        # Incrementally update parquet_info.json
        if self.info_tracker is not None:
            self.info_tracker.register(path)

    def finish(self):
        self._flush()


# ═════════════════════════════════════════════════════════════════════════
#  Dataset processing functions
# ═════════════════════════════════════════════════════════════════════════

def process_spar(json_path: str, writer: ParquetBufferedWriter, need_3d_annotation: bool = False):
    """
    Load SPAR mix JSON → rows.

    The mix JSON maps dataset names to dicts like:
        {
            "dataset_a": {"annotation": "/path/to/anno.jsonl", "root": "/img/root", "repeat_time": 1},
            ...
        }

    Each line in the annotation file is a JSON object with:
        image, conversations, type, id, ...
    """
    print(f"\n[SPAR] Loading mix JSON {json_path}")
    with open(json_path, "r") as f:
        ds_collections = json.load(f)

    total_processed = 0
    total_skipped = 0
    img_stats = ImageStatsCollector("SPAR")

    for ds_name, ds_meta in ds_collections.items():
        # # Skip datasets with 'rxr' in their name
        # if 'rxr' in ds_name:
        #     print(f"\n  [SPAR/{ds_name}] skipped (contains 'rxr')")
        #     continue
        annotation_path = os.path.join("/apdcephfs_303747097/share_303747097/jingfanchen/code/benchmark/SPAR/", ds_meta["annotation"])
        image_root = os.path.join("/apdcephfs_303747097/share_303747097/jingfanchen/code/benchmark/SPAR/", ds_meta["root"])
        repeat_time = ds_meta.get("repeat_time", 1)

        print(f"\n  [SPAR/{ds_name}] annotation={annotation_path}, root={image_root}, repeat={repeat_time}")

        # Read annotation lines
        with open(annotation_path, "r") as f:
            lines = f.readlines()

        # Apply repeat_time
        if isinstance(repeat_time, int) and repeat_time > 1:
            lines = lines * repeat_time
        elif isinstance(repeat_time, (int, float)) and repeat_time < 1:
            lines = lines[: int(len(lines) * repeat_time)]

        skipped = 0
        for line in tqdm(lines, desc=f"  SPAR/{ds_name}"):
            line = line.strip()
            if not line:
                continue
            try:
                data_item = json.loads(line)
            except Exception as e:
                print(f"    [WARN] bad JSON line: {e}")
                skipped += 1
                continue

            row = _spar_item_to_row(
                data_item, 
                image_root, 
                ds_name,
                need_3d_annotation=need_3d_annotation
            )
            if row is None:
                skipped += 1
                continue

            # Record image count statistics
            n_images = len(row["image_list"])
            # Skip samples with more than 5 images
            if n_images > 4:
                skipped += 1
                continue
            img_stats.record(n_images, {
                "dataset": ds_name,
                "id": data_item.get("id", ""),
                "type": data_item.get("type", ""),
                "image_list": row["image_list"],
                "question": row["question"][:200] if row["question"] else "",
            })

            writer.add(row)

        n = len(lines)
        total_processed += n - skipped
        total_skipped += skipped
        print(f"  [SPAR/{ds_name}] processed={n - skipped}, skipped={skipped}")

    print(f"[SPAR] total processed={total_processed}, total skipped={total_skipped}")
    img_stats.print_summary()


def _mindcube_item_to_row(data_item: dict, image_root: str, ds_name: str, index: int = 0) -> Optional[dict]:
    """
    Convert a single MindCube annotation item into the target Parquet row schema.

    Supports two annotation formats:

    1. **qwen_sft** (``training/qwen2.5vl/*.json``) – SPAR-like format:
       - ``images``: list[str]
       - ``conversations``: [{from: "human", value: ...}, {from: "gpt", value: ...}]

    2. **general JSONL** (``prompts/general/*.jsonl``) – original format:
       - ``images``: list[str]
       - ``input_prompt``: str
       - ``gt_answer``: str
    """
    # ── resolve image paths ──
    raw_images = data_item.get("images", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    image_list = [os.path.join(image_root, p) for p in raw_images]

    # ── 读取图片字节 ──
    image_bytes_list: List[bytes] = []
    valid_image_list: List[str] = []
    for img_path in image_list:
        try:
            with open(img_path, "rb") as f:
                image_bytes_list.append(f.read())
            valid_image_list.append(img_path)
        except Exception as e:
            print(f"    [WARN] cannot read image {img_path}: {e}")
            continue

    if not valid_image_list:
        return None

    image_list = valid_image_list
    n_images = len(image_list)

    # ── extract question / answer ──
    # Try conversations format first (qwen_sft), fall back to direct fields
    convs = data_item.get("conversations", [])
    if convs:
        question = ""
        answer = ""
        for turn in convs:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                question = value # .replace('<image>\n', '')
            elif role == "gpt":
                answer = value
    else:
        question = data_item.get("input_prompt", "") or data_item.get("question", "")
        answer = data_item.get("gt_answer", "")

    # ── metadata ──
    meta = {
        "type": data_item.get("type", "unknown"),
        "id": data_item.get("id", ""),
    }
    skip_keys = {"images", "conversations", "input_prompt", "question",
                 "gt_answer", "type", "id"}
    for k, v in data_item.items():
        if k not in skip_keys:
            meta[k] = v

    return {
        "index":           index,
        "question":        question,
        "input_prompt":    question,
        "answer":          answer,
        "scene_name":      "mindcube",
        # "category":        "among",
        "dataset_name":    f"mindcube_{ds_name}",
        "image_path":      image_list,
        # "image_bytes":     image_bytes_list,
        "depth_list":      [],
        # "depth_bytes":     [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_mindcube(
    json_path: str,
    writer: ParquetBufferedWriter,
    mindcube_data_root: str = "/data/spatial_data/reason_data",
    csv_output_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
):
    """
    Load MindCube mix JSON → rows.

    The mix JSON maps dataset names to dicts like::

        {
            "MindCube_train_raw_qa": {
                "root": "mindcube/MindCube/data",
                "annotation": "mindcube/MindCube/data/prompts/training/qwen2.5vl/MindCube_train_raw_qa_qwen_sft.json",
                "repeat_time": 1.0,
                "length": 10000
            },
            ...
        }

    Annotation files can be either:
      - JSON array (``*.json``)  – e.g. qwen_sft training files
      - JSONL (``*.jsonl``)      – one JSON object per line
    """
    print(f"\n[MindCube] Loading mix JSON {json_path}")
    with open(json_path, "r") as f:
        ds_collections = json.load(f)

    total_processed = 0
    total_skipped = 0
    img_stats = ImageStatsCollector("MindCube")
    all_rows_for_csv: List[dict] = []  # Collect rows for CSV export

    for ds_name, ds_meta in ds_collections.items():
        if 'raw_qa' not in ds_name:
            continue
        annotation_path = os.path.join(mindcube_data_root, ds_meta["annotation"])
        image_root = os.path.join(mindcube_data_root, ds_meta["root"])
        repeat_time = ds_meta.get("repeat_time", 1)

        print(f"\n  [MindCube/{ds_name}] annotation={annotation_path}, root={image_root}, repeat={repeat_time}")

        # ── Load data items: JSON array or JSONL ──
        if annotation_path.endswith(".json"):
            with open(annotation_path, "r") as f:
                data_items = json.load(f)
            if not isinstance(data_items, list):
                print(f"    [WARN] Expected JSON array, got {type(data_items)}. Skipping.")
                continue
        else:
            # JSONL: one JSON object per line
            data_items = []
            with open(annotation_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data_items.append(json.loads(line))
                        except Exception as e:
                            print(f"    [WARN] bad JSON line: {e}")

        # ── Apply repeat_time ──
        if isinstance(repeat_time, int) and repeat_time > 1:
            data_items = data_items * repeat_time
        elif isinstance(repeat_time, (int, float)) and repeat_time < 1:
            data_items = data_items[: int(len(data_items) * repeat_time)]

        # ── 全局 shuffle（固定随机种子，保证可复现） ──
        import random as _random
        _rng = _random.Random(42)
        _rng.shuffle(data_items)
        print(f"  [MindCube/{ds_name}] shuffled {len(data_items)} items (seed=42)")

        skipped = 0
        global_idx = 0
        # data_items = data_items[:10]
        for data_item in tqdm(data_items, desc=f"  MindCube/{ds_name}"):
            row = _mindcube_item_to_row(data_item, image_root, ds_name, index=global_idx)
            global_idx += 1
            if row is None:
                skipped += 1
                continue

            # Record image count statistics
            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "dataset": ds_name,
                "id": data_item.get("id", ""),
                "type": data_item.get("type", ""),
                "image_path": row["image_path"],
                "question": row["question"][:200] if row["question"] else "",
            })

            writer.add(row)
            all_rows_for_csv.append(row)

        n = len(data_items)
        total_processed += n - skipped
        total_skipped += skipped
        print(f"  [MindCube/{ds_name}] processed={n - skipped}, skipped={skipped}")

    print(f"[MindCube] total processed={total_processed}, total skipped={total_skipped}")
    img_stats.print_summary()

    # ── Save to CSV and optionally upload to HuggingFace ──
    # if all_rows_for_csv:
    #     _save_csv_and_upload_hf(all_rows_for_csv, csv_output_path, hf_repo, dataset_label="mindcube")


# ═════════════════════════════════════════════════════════════════════════
#  3DThinker-10K
# ═════════════════════════════════════════════════════════════════════════

def _3dthinker_item_to_row(data_item: dict, data_root: str, index: int = 0) -> Optional[dict]:
    """
    将 3DThinker-10K 的一条 JSONL 记录转换为 Parquet 行。

    JSONL 字段:
      - ``mindcube_input``: question（包含 <image> 占位符）
      - ``text_output``:    answer（含 <output_3D> 思维链）
      - ``image_input``:    list[str]，相对路径（如 data/other_all_image_resize/...）
      - ``idx``:            样本 id
    """
    # ── 解析图片路径 ──
    raw_images = data_item.get("image_input", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    # image_input 中的路径形如 "data/other_all_image_resize/..."
    # data_root 应指向包含 "data/" 子目录的父目录，
    # 或者直接指向 other_all_image_resize 的父目录。
    # 这里兼容两种情况：如果 data_root 下直接有 "data/" 则直接拼接，
    # 否则尝试去掉 "data/" 前缀后拼接。
    image_list = []
    for p in raw_images:
        full = os.path.join(data_root, p)
        if not os.path.isfile(full):
            # 尝试去掉 "data/" 前缀
            stripped = p
            if p.startswith("data/"):
                stripped = p[len("data/"):]
            full = os.path.join(data_root, stripped)
        image_list.append(full)

    # ── 读取图片字节并验证 ──
    valid_image_list: List[str] = []
    for img_path in image_list:
        if os.path.isfile(img_path):
            valid_image_list.append(img_path)
        else:
            print(f"    [WARN] cannot find image {img_path}")

    if not valid_image_list:
        return None

    image_list = valid_image_list
    n_images = len(image_list)

    # ── 提取 question / answer ──
    question = data_item.get("mindcube_input", "")
    answer = data_item.get("text_output", "")

    # ── metadata ──
    meta = {
        "type": "3dthinker",
        "id": data_item.get("idx", index),
    }
    skip_keys = {"mindcube_input", "mindcube_output", "text_input",
                 "text_output", "image_input", "answer", "idx"}
    for k, v in data_item.items():
        if k not in skip_keys:
            meta[k] = v

    return {
        "index":           index,
        "question":        question,
        "input_prompt":    question,
        "answer":          answer,
        "scene_name":      "3dthinker",
        "dataset_name":    "3dthinker_10k",
        "image_path":      image_list,
        "depth_list":      [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_3dthinker(
    jsonl_path: str,
    writer: "ParquetBufferedWriter",
    data_root: str,
):
    """
    加载 3DThinker-10K JSONL → Parquet 行。

    参数:
        jsonl_path: data_output3d_begin_10k_resized.jsonl 的路径
        writer:     ParquetBufferedWriter 实例
        data_root:  图片根目录（包含 other_all_image_resize/ 的目录）
    """
    print(f"\n[3DThinker] Loading JSONL {jsonl_path}")
    data_items = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_items.append(json.loads(line))
                except Exception as e:
                    print(f"    [WARN] bad JSON line: {e}")

    print(f"[3DThinker] Loaded {len(data_items)} items")

    img_stats = ImageStatsCollector("3DThinker")
    skipped = 0

    for idx, data_item in enumerate(tqdm(data_items, desc="  3DThinker")):
        row = _3dthinker_item_to_row(data_item, data_root, index=idx)
        if row is None:
            skipped += 1
            continue

        n_images = len(row["image_path"])
        img_stats.record(n_images, {
            "id": data_item.get("idx", idx),
            "image_path": row["image_path"],
            "question": row["question"][:200] if row["question"] else "",
        })

        writer.add(row)

    total = len(data_items)
    print(f"[3DThinker] processed={total - skipped}, skipped={skipped}")
    img_stats.print_summary()


def process_omnispatial(
    dataset_path: str,
    prompt_type: str,
    eval_type: str,
    writer: ParquetBufferedWriter,
):
    """Load OmniSpatial → rows."""
    print(f"\n[OmniSpatial] Loading {dataset_path}")
    ds = OmniSpatialDataset(dataset_path, prompt_type, eval_type)
    skipped = 0
    for idx in tqdm(range(len(ds)), desc="OmniSpatial"):
        item = ds[idx]
        row = _imageonly_item_to_row(item)
        if row is None:
            skipped += 1
            continue
        writer.add(row)
    print(f"[OmniSpatial] processed={len(ds)}, skipped={skipped}")


def process_ost(
    data_file: str,
    image_root: str,
    writer: ParquetBufferedWriter,
):
    """Load OST-Bench → rows."""
    print(f"\n[OST-Bench] Loading {data_file}")
    ds = OSTBenchDataset(data_file, image_root)
    skipped = 0
    for idx in tqdm(range(len(ds)), desc="OST-Bench"):
        item = ds[idx]
        row = _imageonly_item_to_row(item)
        if row is None:
            skipped += 1
            continue
        writer.add(row)
    print(f"[OST-Bench] processed={len(ds)}, skipped={skipped}")


def _sensenova_si_item_to_row(data_item: dict, image_root: str, task_type: str) -> Optional[dict]:
    """
    Convert a single SenseNova-SI-800K annotation line (JSON object) into
    the target Parquet row schema.

    Each line has:
      - ``id``: int – unique identifier
      - ``conversations``: [{from, value}, ...] – human/gpt turns
      - ``image``: list[str] – relative image paths (e.g. 'images/000/025035.jpg')

    The ``<image>`` placeholders in conversation text mark where images
    are inserted; the number of placeholders matches len(image).
    """
    # ── resolve image paths ──
    raw_images = data_item.get("image", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    image_list = [os.path.join(image_root, p) for p in raw_images]

    # ── extract question / answer from conversations ──
    convs = data_item.get("conversations", [])
    question = ""
    answer = ""
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            question = value
        elif role == "gpt":
            answer = value

    # ── metadata ──
    if task_type == 'geo':
        meta = {
            "type": "sensenova_si",
            "task": "geo",
            "id": data_item.get("id", ""),
        }
    elif task_type == 'und':
        meta = {
            "type": "sensenova_si",
            "task": "und",
            "id": data_item.get("id", ""),
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    n_images = len(image_list)

    return {
        "question":        question,
        "answer":          answer,
        "scene_name":      "sensenova_si",
        "dataset_name":    "sensenova_si",
        "image_path":      image_list,
        "depth_list":      [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_sensenova_si(
    jsonl_path: str,
    image_root: str,
    writer: ParquetBufferedWriter = None,
    min_images_per_sample: int = 1,
    max_images_per_sample: int = 8,
    task = 'geo',
):
    """
    Load SenseNova-SI-800K JSONL → rows.

    Each sample is derived into **two** tasks (geo and und), written to
    separate Parquet writers.

    Args:
        jsonl_path: Path to SenseNova-SI-800K.jsonl
        image_root: Root directory for resolving relative image paths
                    (the directory containing the 'images/' folder)
        geo_writer: ParquetBufferedWriter for geo task
        und_writer: ParquetBufferedWriter for und task
        min_images_per_sample: Skip samples with fewer images than this
        max_images_per_sample: Skip samples with more images than this
    """
    print(f"\n[SenseNova-SI] Loading {jsonl_path}")

    # Count lines first for progress bar
    with open(jsonl_path, "r") as f:
        total_lines = sum(1 for _ in f)
    print(f"  Total lines: {total_lines}")

    img_stats = ImageStatsCollector("SenseNova-SI")
    total_processed = 0
    total_skipped = 0

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, total=total_lines, desc="SenseNova-SI"):
            line = line.strip()
            if not line:
                continue
            try:
                data_item = json.loads(line)
            except Exception as e:
                print(f"    [WARN] bad JSON line: {e}")
                total_skipped += 1
                continue

            row = _sensenova_si_item_to_row(data_item, image_root, task_type=task)

            if row is None:
                total_skipped += 1
                continue

            n_images = len(row["image_path"])
            if n_images > max_images_per_sample:
                total_skipped += 1
                continue
            if n_images < min_images_per_sample:
                total_skipped += 1
                continue
            

            img_stats.record(n_images, {
                "id": data_item.get("id", ""),
                "image_path": row["image_path"],
                "question": row["question"][:200] if row["question"] else "",
            })
            writer.add(row)
            total_processed += 1

    print(f"[SenseNova-SI] task={task} processed={total_processed}, skipped={total_skipped}")
    img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  RE10K (RealEstate10K) – multi-view scene dataset
# ═════════════════════════════════════════════════════════════════════════

def _re10k_scene_to_row(scene_dir: str, scene_name: str) -> Optional[dict]:
    """
    Convert a single RE10K scene folder into the target Parquet row schema.

    Each scene folder contains multiple PNG frames named by timestamps
    (e.g. ``52553000.png``).  All frames of a scene are packed into one row
    as a multi-view image list.

    Args:
        scene_dir:  Absolute path to the scene folder.
        scene_name: Name of the scene (folder basename, e.g. '0000cc6d8b108390').

    Returns:
        A dict matching ``ARROW_SCHEMA``, or ``None`` if no valid images found.
    """
    # Collect all PNG files sorted by timestamp
    image_files = sorted(
        f for f in os.listdir(scene_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    if not image_files:
        return None

    image_list = [os.path.join(scene_name, f) for f in image_files]
    n_images = len(image_list)

    # RE10K is a pure multi-view reconstruction dataset:
    # no text question/answer, no depth, no pose in this raw format.
    meta = {
        "type": "re10k",
        "id": scene_name,
        "n_frames": n_images,
    }

    return {
        "question":        "",
        "answer":          "",
        "scene_name":      scene_name,
        "dataset_name":    "re10k",
        "image_path":      image_list,
        "depth_list":      [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_re10k(
    data_root: str,
    writer: ParquetBufferedWriter,
    min_frames: int = 2,
    max_frames: int = 0,
):
    """
    Scan all scene folders under ``data_root`` and convert each scene
    into one Parquet row.

    Args:
        data_root:  Path to the RE10K snapshot directory containing
                    scene folders (e.g. ``0000cc6d8b108390/``).
        writer:     Parquet writer to flush rows into.
        min_frames: Skip scenes with fewer frames than this (default 2).
        max_frames: Skip scenes with more frames than this (0 = no limit).
    """
    print(f"\n[RE10K] Scanning scenes under {data_root}")

    # Enumerate scene directories
    scene_names = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    print(f"  Found {len(scene_names)} scene folders")

    img_stats = ImageStatsCollector("RE10K")
    total_processed = 0
    total_skipped = 0

    for scene_name in tqdm(scene_names, desc="RE10K scenes"):
        scene_dir = os.path.join(data_root, scene_name)
        row = _re10k_scene_to_row(scene_dir, scene_name)
        if row is None:
            total_skipped += 1
            continue

        n_images = len(row["image_path"])

        # Apply frame count filters
        if n_images < min_frames:
            total_skipped += 1
            continue
        if max_frames > 0 and n_images > max_frames:
            total_skipped += 1
            continue

        img_stats.record(n_images, {
            "scene": scene_name,
            "n_frames": n_images,
            "first_image": row["image_path"][0] if row["image_path"] else "",
        })

        writer.add(row)
        total_processed += 1

    print(f"[RE10K] processed={total_processed}, skipped={total_skipped}")
    img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  ARKitScenes – multi-view indoor scene dataset
# ═════════════════════════════════════════════════════════════════════════

def _parse_traj_file(traj_path: str) -> Dict[float, np.ndarray]:
    """
    Parse an ARKitScenes ``lowres_wide.traj`` file.

    Each line has 7 values:
        timestamp  qx qy qz  tx ty tz
    where (qx, qy, qz) is a *rotation vector* (axis-angle, Rodrigues) and
    (tx, ty, tz) is the translation.

    Returns:
        Dict mapping timestamp → 4×4 extrinsic matrix (camera-to-world).
    """
    from scipy.spatial.transform import Rotation

    poses: Dict[float, np.ndarray] = {}
    with open(traj_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            ts = float(parts[0])
            rotvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            t = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            R = Rotation.from_rotvec(rotvec).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = R
            mat[:3, 3] = t
            poses[ts] = mat
    return poses


def _parse_pincam(pincam_path: str) -> np.ndarray:
    """
    Parse an ARKitScenes ``.pincam`` intrinsic file.

    Format: ``w h fx fy cx cy``

    Returns:
        4×4 intrinsic matrix (flattened to 16-element list later).
    """
    with open(pincam_path, "r") as f:
        parts = f.readline().strip().split()
    w, h = int(parts[0]), int(parts[1])
    fx, fy, cx, cy = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
    K = np.eye(4)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _match_timestamp(ts_query: float, ts_dict: Dict[float, Any], tol: float = 0.05) -> Optional[float]:
    """
    Find the closest timestamp key in *ts_dict* within *tol* seconds.

    Returns the matched key, or ``None`` if no match within tolerance.
    """
    best_key = None
    best_diff = float("inf")
    for k in ts_dict:
        diff = abs(k - ts_query)
        if diff < best_diff:
            best_diff = diff
            best_key = k
    if best_diff <= tol:
        return best_key
    return None


def _arkitscenes_scene_to_rows(
    scene_dir: str,
    scene_id: str,
    max_frames: int = 0,
    frame_stride: int = 1,
    need_3d_annotation: bool = True,
) -> Optional[dict]:
    """
    Convert a single ARKitScenes scene folder into the target Parquet row.

    Uses ``lowres_wide`` images, ``lowres_depth`` depth maps,
    ``lowres_wide.traj`` for camera poses, and ``lowres_wide_intrinsics``
    for per-frame camera intrinsics.

    Args:
        scene_dir:  Absolute path to the scene folder.
        scene_id:   Scene identifier (folder basename).
        max_frames: Maximum number of frames to keep (0 = all).
        frame_stride: Sub-sample every N-th frame (default 1 = keep all).
        need_3d_annotation: Whether to load depth/pose/intrinsic data.

    Returns:
        A dict matching ``ARROW_SCHEMA``, or ``None`` on failure.
    """
    # ── Locate image directory (extracted or need to skip if zipped) ──
    lowres_wide_dir = os.path.join(scene_dir, "lowres_wide")
    if not os.path.isdir(lowres_wide_dir):
        # Images not extracted – skip this scene
        return None

    # ── Collect and sort image files by timestamp ──
    image_files = sorted(
        f for f in os.listdir(lowres_wide_dir)
        if f.lower().endswith(".png")
    )
    if not image_files:
        return None

    # Extract timestamps from filenames: {scene_id}_{timestamp}.png
    def _extract_ts(fname: str) -> float:
        # e.g. "40753679_6789.499.png" → 6789.499
        stem = fname.rsplit(".", 1)[0]  # "40753679_6789.499"
        ts_str = stem.split("_", 1)[1]  # "6789.499"
        return float(ts_str)

    ts_file_pairs = []
    for f in image_files:
        try:
            ts = _extract_ts(f)
            ts_file_pairs.append((ts, f))
        except (ValueError, IndexError):
            continue
    ts_file_pairs.sort(key=lambda x: x[0])

    # Apply stride
    if frame_stride > 1:
        ts_file_pairs = ts_file_pairs[::frame_stride]

    # Apply max_frames
    if max_frames > 0 and len(ts_file_pairs) > max_frames:
        # Uniformly sub-sample
        indices = np.linspace(0, len(ts_file_pairs) - 1, max_frames, dtype=int)
        ts_file_pairs = [ts_file_pairs[i] for i in indices]

    if not ts_file_pairs:
        return None

    image_list = [os.path.join(lowres_wide_dir, f) for _, f in ts_file_pairs]
    n_images = len(image_list)

    # ── 3D annotation ──
    if need_3d_annotation:
        # -- Poses from trajectory --
        traj_path = os.path.join(scene_dir, "lowres_wide.traj")
        if os.path.isfile(traj_path):
            traj_poses = _parse_traj_file(traj_path)
            poses = []
            for ts, _ in ts_file_pairs:
                matched_ts = _match_timestamp(ts, traj_poses)
                if matched_ts is not None:
                    poses.append(traj_poses[matched_ts].flatten().tolist())
                else:
                    poses.append(_IDENTITY_4x4)
        else:
            poses = [_IDENTITY_4x4] * n_images

        # -- Depth maps --
        lowres_depth_dir = os.path.join(scene_dir, "lowres_depth")
        if os.path.isdir(lowres_depth_dir):
            depth_list = []
            for ts, img_f in ts_file_pairs:
                # Depth file uses same naming: {scene_id}_{timestamp}.png
                depth_fname = img_f  # same filename convention
                depth_path = os.path.join(lowres_depth_dir, depth_fname)
                if os.path.isfile(depth_path):
                    depth_list.append(depth_path)
                else:
                    depth_list.append("")  # missing depth frame
        else:
            depth_list = []

        # -- Intrinsics (use first available .pincam as representative) --
        intrinsics_dir = os.path.join(scene_dir, "lowres_wide_intrinsics")
        intrinsic = _IDENTITY_4x4
        if os.path.isdir(intrinsics_dir):
            # Try to match the first frame's timestamp
            first_ts, first_f = ts_file_pairs[0]
            pincam_name = first_f.rsplit(".", 1)[0] + ".pincam"  # e.g. 40753679_6789.499.pincam
            pincam_path = os.path.join(intrinsics_dir, pincam_name)
            if os.path.isfile(pincam_path):
                try:
                    K = _parse_pincam(pincam_path)
                    intrinsic = K.flatten().tolist()
                except Exception:
                    pass

        depth_intrinsic = intrinsic  # depth uses same camera
    else:
        poses = [_IDENTITY_4x4] * n_images
        depth_list = []
        intrinsic = _IDENTITY_4x4
        depth_intrinsic = _IDENTITY_4x4

    meta = {
        "type": "arkitscenes",
        "id": scene_id,
        "n_frames": n_images,
    }

    return {
        "question":        "",
        "answer":          "",
        "scene_name":      "arkitscenes",
        "dataset_name":    "arkitscenes",
        "image_path":      image_list,
        "depth_list":      depth_list,
        "poses":           poses,
        "intrinsic":       intrinsic,
        "depth_intrinsic": depth_intrinsic,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_arkitscenes(
    data_root: str,
    writer: ParquetBufferedWriter,
    max_frames: int = 0,
    frame_stride: int = 1,
    need_3d_annotation: bool = True,
):
    """
    Scan all scene folders under ``data_root`` and convert each extracted
    ARKitScenes scene into one Parquet row.

    Scenes that are only stored as zip files (not yet extracted) are
    skipped automatically.

    Args:
        data_root:   Path to ARKitScenes Training directory
                     (e.g. ``.../ar_raw_all/raw/Training``).
        writer:      Parquet writer to flush rows into.
        max_frames:  Maximum frames per scene (0 = no limit).
        frame_stride: Sub-sample every N-th frame (default 1 = keep all).
        need_3d_annotation: Whether to load depth/pose/intrinsic data.
    """
    print(f"\n[ARKitScenes] Scanning scenes under {data_root}")

    scene_ids = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    print(f"  Found {len(scene_ids)} scene folders")

    img_stats = ImageStatsCollector("ARKitScenes")
    total_processed = 0
    total_skipped = 0

    for scene_id in tqdm(scene_ids, desc="ARKitScenes scenes"):
        scene_dir = os.path.join(data_root, scene_id)
        row = _arkitscenes_scene_to_rows(
            scene_dir,
            scene_id,
            max_frames=max_frames,
            frame_stride=frame_stride,
            need_3d_annotation=need_3d_annotation,
        )
        if row is None:
            total_skipped += 1
            continue

        n_images = len(row["image_path"])
        img_stats.record(n_images, {
            "scene": scene_id,
            "n_frames": n_images,
            "first_image": row["image_path"][0] if row["image_path"] else "",
        })

        writer.add(row)
        total_processed += 1

    print(f"[ARKitScenes] processed={total_processed}, skipped={total_skipped}")
    img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  SPLD (SpatialLadder-26k) – spatial reasoning + grounding (image-only)
# ═════════════════════════════════════════════════════════════════════════

def _discover_spld_jsonl_files(data_root: str) -> Dict[str, str]:
    """
    在 SpatialLadder HuggingFace 缓存目录中自动发现 JSONL 数据文件。

    HuggingFace datasets 缓存将 JSONL 文件存储为 blob（无扩展名），
    因此我们通过读取每个 blob 的第一行来识别文件类型。

    Returns:
        Dict mapping logical name → absolute blob path, e.g.:
        {
            "spatial": "/path/to/blob_hash_1",
            "grounding": "/path/to/blob_hash_2",
            "coldstart": "/path/to/blob_hash_3",
        }
    """
    blobs_dir = os.path.join(data_root, "..", "..", "blobs")
    blobs_dir = os.path.normpath(blobs_dir)

    # 如果 blobs 目录不存在，尝试直接在 data_root 下查找 .jsonl 文件
    discovered: Dict[str, str] = {}

    if os.path.isdir(blobs_dir):
        for fname in os.listdir(blobs_dir):
            fpath = os.path.join(blobs_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # 跳过太小（< 100KB）或太大（> 50MB，可能是 images.zip）的文件
            fsize = os.path.getsize(fpath)
            if fsize < 100_000 or fsize > 50_000_000:
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    continue
                obj = json.loads(first_line)
                # 根据字段特征区分文件类型
                if "options" in obj and "question_type" in obj:
                    discovered["spatial"] = fpath
                elif "bbox_2d" in str(obj.get("answer", "")):
                    discovered["grounding"] = fpath
                elif "prompt" in obj and "solution" in obj:
                    discovered["coldstart"] = fpath
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                continue

    # 也尝试在 data_root 下查找标准命名的 .jsonl 文件
    for name, pattern in [
        ("spatial", "spld_spatial_data.jsonl"),
        ("grounding", "spld_grounding_data.jsonl"),
        ("coldstart", "spld_coldstart_data.jsonl"),
    ]:
        candidate = os.path.join(data_root, pattern)
        if os.path.isfile(candidate) and name not in discovered:
            discovered[name] = candidate

    return discovered


def _spld_spatial_item_to_row(
    data_item: dict,
    image_root: str,
) -> Optional[dict]:
    """
    将一条 SpatialLadder spatial JSONL 记录转换为目标 Parquet 行。

    字段格式:
      - question_id: int
      - question: str
      - options: list[str] 或 null
      - answer: str  (选项字母如 "A"，或数值如 "3")
      - image: list[str]  (相对路径如 "scene0000_00/820.jpg")
      - question_type: str  (如 "relative direction", "object count" 等)
      - data_type: str  ("single_image", "multi_view", "video")
    """
    raw_images = data_item.get("image", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    image_list = [os.path.join(image_root, p) for p in raw_images]
    n_images = len(image_list)

    question = data_item.get("question", "")
    options = data_item.get("options")
    answer = data_item.get("answer", "")

    # 将选项拼接到问题中（如果有的话）
    if options:
        question_with_options = question + "\n" + "\n".join(options)
    else:
        question_with_options = question

    data_type = data_item.get("data_type", "unknown")
    question_type = data_item.get("question_type", "unknown")

    meta = {
        "type": "spld_spatial",
        "id": data_item.get("question_id", ""),
        "data_type": data_type,
        "question_type": question_type,
    }
    if options:
        meta["options"] = options

    return {
        "question":        question_with_options,
        "answer":          str(answer),
        "scene_name":      "scannet",
        "dataset_name":    f"spld_spatial_{data_type}",
        "image_path":      image_list,
        "depth_list":      [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def _spld_grounding_item_to_row(
    data_item: dict,
    image_root: str,
) -> Optional[dict]:
    """
    将一条 SpatialLadder grounding JSONL 记录转换为目标 Parquet 行。

    字段格式:
      - question_id: int
      - question: str
      - answer: list[dict]  每个 dict 含 bbox_2d (list[int]) 和 label (str)
      - image: list[str]
      - data_type: str
    """
    raw_images = data_item.get("image", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    image_list = [os.path.join(image_root, p) for p in raw_images]
    n_images = len(image_list)

    question = data_item.get("question", "")
    raw_answer = data_item.get("answer", [])

    # 将 grounding answer 序列化为字符串
    if isinstance(raw_answer, list):
        answer_str = json.dumps(raw_answer, ensure_ascii=False)
    else:
        answer_str = str(raw_answer)

    data_type = data_item.get("data_type", "unknown")

    meta = {
        "type": "spld_grounding",
        "id": data_item.get("question_id", ""),
        "data_type": data_type,
    }

    return {
        "question":        question,
        "answer":          answer_str,
        "scene_name":      "scannet",
        "dataset_name":    "spld_grounding",
        "image_path":      image_list,
        "depth_list":      [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def _spld_coldstart_item_to_row(
    data_item: dict,
    image_root: str,
) -> Optional[dict]:
    """
    将一条 SpatialLadder coldstart JSONL 记录转换为目标 Parquet 行。

    字段格式:
      - question_id: int
      - prompt: str  (包含 CoT 提示的完整问题)
      - solution: str  (包含 <think>...</think><answer>...</answer> 的回答)
      - image: list[str]
      - data_type: str
    """
    raw_images = data_item.get("image", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images:
        return None

    image_list = [os.path.join(image_root, p) for p in raw_images]
    n_images = len(image_list)

    question = data_item.get("prompt", "")
    answer = data_item.get("solution", "")
    data_type = data_item.get("data_type", "unknown")

    meta = {
        "type": "spld_coldstart",
        "id": data_item.get("question_id", ""),
        "data_type": data_type,
    }

    return {
        "question":        question,
        "answer":          answer,
        "scene_name":      "scannet",
        "dataset_name":    "spld_coldstart",
        "image_path":      image_list,
        "depth_list":      [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_spld(
    data_root: str,
    writer: ParquetBufferedWriter,
):
    """
    加载 SpatialLadder-26k 数据集并转换为 Parquet 行。

    自动发现 data_root 对应的 HuggingFace 缓存中的 JSONL 文件
    （spatial / grounding / coldstart），并将所有样本写入同一个 writer。

    图片路径为相对路径（如 "scene0000_00/820.jpg"），以 data_root 为根目录拼接。

    Args:
        data_root:  SpatialLadder-26k 的 snapshot 目录路径
                    （包含 scene 文件夹和图片）。
        writer:     ParquetBufferedWriter 用于写入 Parquet 文件。
    """
    print(f"\n[SPLD] Loading SpatialLadder-26k from {data_root}")

    # 自动发现 JSONL 文件
    jsonl_files = _discover_spld_jsonl_files(data_root)
    if not jsonl_files:
        print("  [ERROR] No SPLD JSONL files found. "
              "Make sure the data_root points to the HuggingFace snapshot directory.")
        return

    print(f"  Discovered JSONL files: {list(jsonl_files.keys())}")
    for name, path in jsonl_files.items():
        print(f"    {name}: {path}")

    img_stats = ImageStatsCollector("SPLD")
    total_processed = 0
    total_skipped = 0

    # 图片根目录就是 data_root（scene 文件夹在其下）
    image_root = data_root

    # ── 处理 spatial 数据 ──
    if "spatial" in jsonl_files:
        spatial_path = jsonl_files["spatial"]
        print(f"\n  [SPLD/spatial] Loading {spatial_path}")
        with open(spatial_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"  [SPLD/spatial] {len(lines)} lines")

        skipped = 0
        for line in tqdm(lines, desc="  SPLD/spatial"):
            line = line.strip()
            if not line:
                continue
            try:
                data_item = json.loads(line)
            except Exception as e:
                print(f"    [WARN] bad JSON line: {e}")
                skipped += 1
                continue

            row = _spld_spatial_item_to_row(data_item, image_root)
            if row is None:
                skipped += 1
                continue

            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "id": data_item.get("question_id", ""),
                "data_type": data_item.get("data_type", ""),
                "question_type": data_item.get("question_type", ""),
                "question": row["question"][:200],
            })
            writer.add(row)

        processed = len(lines) - skipped
        total_processed += processed
        total_skipped += skipped
        print(f"  [SPLD/spatial] processed={processed}, skipped={skipped}")

    # ── 处理 grounding 数据 ──
    if "grounding" in jsonl_files:
        grounding_path = jsonl_files["grounding"]
        print(f"\n  [SPLD/grounding] Loading {grounding_path}")
        with open(grounding_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"  [SPLD/grounding] {len(lines)} lines")

        skipped = 0
        for line in tqdm(lines, desc="  SPLD/grounding"):
            line = line.strip()
            if not line:
                continue
            try:
                data_item = json.loads(line)
            except Exception as e:
                print(f"    [WARN] bad JSON line: {e}")
                skipped += 1
                continue

            row = _spld_grounding_item_to_row(data_item, image_root)
            if row is None:
                skipped += 1
                continue

            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "id": data_item.get("question_id", ""),
                "data_type": data_item.get("data_type", ""),
                "question": row["question"][:200],
            })
            writer.add(row)

        processed = len(lines) - skipped
        total_processed += processed
        total_skipped += skipped
        print(f"  [SPLD/grounding] processed={processed}, skipped={skipped}")

    # ── 处理 coldstart 数据 ──
    if "coldstart" in jsonl_files:
        coldstart_path = jsonl_files["coldstart"]
        print(f"\n  [SPLD/coldstart] Loading {coldstart_path}")
        with open(coldstart_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"  [SPLD/coldstart] {len(lines)} lines")

        skipped = 0
        for line in tqdm(lines, desc="  SPLD/coldstart"):
            line = line.strip()
            if not line:
                continue
            try:
                data_item = json.loads(line)
            except Exception as e:
                print(f"    [WARN] bad JSON line: {e}")
                skipped += 1
                continue

            row = _spld_coldstart_item_to_row(data_item, image_root)
            if row is None:
                skipped += 1
                continue

            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "id": data_item.get("question_id", ""),
                "data_type": data_item.get("data_type", ""),
                "question": row["question"][:200],
            })
            writer.add(row)

        processed = len(lines) - skipped
        total_processed += processed
        total_skipped += skipped
        print(f"  [SPLD/coldstart] processed={processed}, skipped={skipped}")

    print(f"[SPLD] total processed={total_processed}, total skipped={total_skipped}")
    img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  VGLLM – VG-LLM-Data series (multiple scene QA / grounding datasets)
# ═════════════════════════════════════════════════════════════════════════

# 从文件名推断子数据集名称
_VGLLM_FILENAME_TO_DATASET = {
    "scan2cap":    "vgllm_scan2cap",
    "scanrefer":   "vgllm_scanrefer",
    "scannet_det": "vgllm_scannet_det",
    "sqa3d":       "vgllm_sqa3d",
    "spar_234k":   "vgllm_spar_234k",
    "spar_7m":     "vgllm_spar_7m",
    "llava_hound": "vgllm_llava_hound",
}


def _infer_vgllm_dataset_name(filepath: str) -> str:
    """根据文件名推断 VGLLM 子数据集名称。"""
    basename = os.path.basename(filepath).lower()
    for key, ds_name in _VGLLM_FILENAME_TO_DATASET.items():
        if key in basename:
            return ds_name
    # 回退：使用文件名（去掉扩展名）
    stem = os.path.splitext(basename)[0]
    return f"vgllm_{stem}"


def _vgllm_item_to_row(
    data_item: dict,
    data_root: str,
    dataset_name: str,
) -> Optional[dict]:
    """
    将一条 VG-LLM 格式的标注转换为目标 Parquet 行。

    VG-LLM 数据集统一使用 ``conversations`` 字段（human/gpt 对话格式），
    图片路径通过 ``images`` 字段（列表）或 ``video`` 字段（目录路径）提供。

    支持的子数据集：
      - scan2cap:    3D 场景描述 (images + input_box/gt_box)
      - scanrefer:   3D 视觉定位 (images + gt_bbox)
      - scannet_det: 3D 目标检测 (images + boxes)
      - sqa3d:       3D 场景问答 (images + situation/question/answer)
      - spar_234k:   空间推理 (images + spar_info)
      - spar_7m:     空间推理 JSONL (images + spar_info)
      - llava_hound: 视频理解 (video 目录 + data_path)

    Args:
        data_item:    单条标注数据。
        data_root:    数据根目录，用于拼接相对路径。
        dataset_name: 子数据集名称（如 "vgllm_scan2cap"）。

    Returns:
        符合 ``ARROW_SCHEMA`` 的 dict，或 ``None``（无有效图片时）。
    """
    # ── 解析图片路径 ──
    image_list = []

    if "images" in data_item:
        # images 字段：相对路径列表
        raw_images = data_item["images"]
        if isinstance(raw_images, str):
            raw_images = [raw_images]
        # 拼接 data_root；如果有 data_path 字段则用它作为中间路径
        # data_path = data_item.get("data_path", "")
        for p in raw_images:
            if 'spar' in dataset_name:
                p = p.replace("spar/", "")
            elif 'llava' in dataset_name:
                p = p.replace('llava_hound/frames/', '')
            else:
                p = p.replace('scannet/posed_images/', '')
            if os.path.isabs(p):
                image_list.append(p)
            else:
                image_list.append(os.path.join(data_root, p))

    elif "video" in data_item:
        # video 字段：帧目录的相对路径（如 "llava_hound/frames/23230678"）
        video_rel = data_item["video"]
        data_path = data_item.get("data_path", "")
        if data_path:
            video_dir = os.path.join(data_root, data_path, video_rel)
        else:
            video_dir = os.path.join(data_root, video_rel)

        if os.path.isdir(video_dir):
            frames = sorted(
                f for f in os.listdir(video_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            image_list = [os.path.join(video_dir, f) for f in frames]
        else:
            # 目录不存在，跳过
            return None

    if not image_list:
        return None

    # ── 读取图片字节 ──
    image_bytes_list: List[bytes] = []
    valid_image_list: List[str] = []
    for img_path in image_list:
        try:
            with open(img_path, "rb") as f:
                image_bytes_list.append(f.read())
            valid_image_list.append(img_path)
        except Exception as e:
            # 图片读取失败，跳过该图片（不中断整条样本）
            print(f"    [WARN] cannot read image {img_path}: {e}")
            continue

    if not valid_image_list:
        return None

    image_list = valid_image_list
    n_images = len(image_list)

    # ── 提取 question / answer ──
    convs = data_item.get("conversations", [])
    question = ""
    answer = ""
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            question = value
        elif role == "gpt":
            answer = value

    # ── 构建 metadata ──
    meta: Dict[str, Any] = {
        "type": dataset_name,
        "id": data_item.get("id", data_item.get("qid", "")),
    }

    # 保留各子数据集的特殊字段
    if dataset_name == 'vgllm_scan2cap':
        metadata = {
            "original_metadata": data_item["metadata"],
            "input_box": data_item["input_box"],
            "gt_box": data_item["gt_box"],
            "iou": data_item["iou"],
        }
    elif dataset_name == 'vgllm_scanrefer':
        metadata = {
            "original_metadata": data_item["metadata"],
            "gt_bbox": data_item["gt_bbox"],
            "target": data_item["target"],
        }
    elif dataset_name == 'vgllm_scannet_det':
        metadata = {
            "boxes": data_item["boxes"],
        }
    elif dataset_name == 'vgllm_sqa3d':
        metadata = {
            "situation": data_item["situation"],
            "question_type": data_item["question_type"],
        }
    elif dataset_name == 'vgllm_spar_234k' or dataset_name == 'vgllm_spar_7m':
        spar_info = data_item["spar_info"]
        if isinstance(spar_info, str):
            spar_info = json.loads(spar_info)
        metadata = {
            "spar_info": spar_info,
            "tag": data_item['tag']
        }
    meta["metadata"] = metadata

    # 推断 scene_name
    scene_name = "scannet"  # 默认
    first_img = image_list[0] if image_list else ""
    if "llava_hound" in first_img or "llava_hound" in dataset_name:
        scene_name = "llava_hound"
    elif "scannet" in first_img:
        scene_name = "scannet"

    return {
        "question":        question,
        "answer":          answer,
        "scene_name":      scene_name,
        "dataset_name":    dataset_name,
        "image_path":      image_list,
        # "image_bytes":     image_bytes_list,
        # "depth_bytes":     [],
        "poses":           [_IDENTITY_4x4] * n_images,
        "intrinsic":       _IDENTITY_4x4,
        "depth_intrinsic": _IDENTITY_4x4,
        "metadata":        _ensure_metadata_str(meta),
    }


def process_mixed_mindcube_vgllm(
    mindcube_json_data_dir_pairs: List[Tuple[str, str, float]],
    vgllm_json_data_dir_pairs: List[Tuple[str, str, float]],
    output_dir: str,
    rows_per_file: int,
    rows_per_row_group: int,
    shuffle_seed: int = 42,
    num_workers: int = 16,
    chunk_size: int = 1024,
    max_samples: Optional[int] = None,
):
    """
    混合处理 MindCube 和 VGLLM 数据集，在 annotation 层面混合后写入同一组 Parquet 文件。

    核心思路：
      1. 分别加载 MindCube 和 VGLLM 的 annotation（JSON/JSONL），得到带标签的 item 列表
      2. 每个数据源通过 repeat_time 控制重复/采样倍率
      3. 全局 shuffle 后，统一调用各自的 _item_to_row 转换并写入同一个 ParquetBufferedWriter

    Args:
        mindcube_json_data_dir_pairs: MindCube (json_path, data_dir, repeat_time) 元组列表
        vgllm_json_data_dir_pairs:    VGLLM (json_path, data_dir, repeat_time) 元组列表
        output_dir:               输出目录
        rows_per_file:            每个 Parquet 文件的最大行数
        rows_per_row_group:       每个 row group 的最大行数
        shuffle_seed:             随机种子
        num_workers:              并行线程数
        chunk_size:               并行处理的批大小
        max_samples:              最大总样本数（None 表示不限制）
    """
    import random as _random
    rng = _random.Random(shuffle_seed)

    # ═══════════════════════════════════════════════════════════════════
    #  第一步：加载所有 annotation，标记来源
    # ═══════════════════════════════════════════════════════════════════

    def _load_json_or_jsonl(path):
        """加载 JSON 或 JSONL 文件，返回 list[dict]。"""
        items = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception as e:
                        print(f"    [WARN] bad JSON line: {e}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # MindCube mix JSON: {ds_name: {annotation, root, ...}, ...}
                items = data
            else:
                print(f"    [WARN] Unexpected JSON type: {type(data)}")
        return items

    def _apply_repeat_time(data_items, repeat_time):
        """根据 repeat_time 重复或截断数据。"""
        if repeat_time >= 2 and int(repeat_time) == repeat_time:
            return data_items * int(repeat_time)
        elif repeat_time > 1:
            # 非整数 > 1，例如 1.5 → 全量 + 50% 随机采样
            full_copies = int(repeat_time)
            frac = repeat_time - full_copies
            result = data_items * full_copies
            extra = int(len(data_items) * frac)
            result += data_items[:extra]
            return result
        elif 0 < repeat_time < 1:
            return data_items[:int(len(data_items) * repeat_time)]
        else:
            return data_items

    # ── 加载 MindCube annotations ──
    mindcube_items: List[Tuple[str, dict, str]] = []  # (source_type, data_item, extra_info)
    for mc_json_path, mc_data_root, mc_repeat_time in mindcube_json_data_dir_pairs:
        mc_data_root = mc_data_root.replace('<', '').replace('>', '')
        print(f"\n[Mixed] 加载 MindCube annotations: {mc_json_path} (data_root={mc_data_root}, repeat={mc_repeat_time})")

        loaded = _load_json_or_jsonl(mc_json_path)

        if isinstance(loaded, dict):
            # MindCube mix JSON 格式: {ds_name: {annotation, root, ...}, ...}
            for ds_name, ds_meta in loaded.items():
                if 'raw_qa' not in ds_name:
                    continue
                annotation_path = os.path.join(mc_data_root, ds_meta["annotation"])
                image_root = os.path.join(mc_data_root, ds_meta["root"])

                print(f"  [MindCube/{ds_name}] annotation={annotation_path}")

                if annotation_path.endswith(".json"):
                    with open(annotation_path, "r") as f:
                        data_items = json.load(f)
                    if not isinstance(data_items, list):
                        print(f"    [WARN] Expected JSON array, got {type(data_items)}. Skipping.")
                        continue
                else:
                    data_items = []
                    with open(annotation_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data_items.append(json.loads(line))
                                except Exception as e:
                                    print(f"    [WARN] bad JSON line: {e}")

                # 应用外部传入的 repeat_time
                data_items = _apply_repeat_time(data_items, mc_repeat_time)

                for item in data_items:
                    mindcube_items.append(("mindcube", item, f"{ds_name}|{image_root}"))
        elif isinstance(loaded, list):
            # 直接是 annotation 列表（JSONL 或 JSON array）
            ds_name = os.path.splitext(os.path.basename(mc_json_path))[0]
            data_items = _apply_repeat_time(loaded, mc_repeat_time)
            for item in data_items:
                mindcube_items.append(("mindcube", item, f"{ds_name}|{mc_data_root}"))

    print(f"  [MindCube] 共加载 {len(mindcube_items)} 条 annotation")

    # ── 加载 VGLLM annotations ──
    vgllm_items: List[Tuple[str, dict, str]] = []  # (source_type, data_item, extra_info)
    for json_path, data_root, vg_repeat_time in vgllm_json_data_dir_pairs:
        dataset_name = _infer_vgllm_dataset_name(json_path)
        data_root = data_root.replace('<', '').replace('>', '')
        print(f"\n  [VGLLM/{dataset_name}] 加载 {json_path} (repeat={vg_repeat_time})")

        is_jsonl = json_path.endswith(".jsonl")
        if is_jsonl:
            data_list = []
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data_list.append(json.loads(line))
                    except Exception as e:
                        print(f"    [WARN] bad JSON line: {e}")
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)

        # 应用外部传入的 repeat_time
        data_list = _apply_repeat_time(data_list, vg_repeat_time)

        for item in data_list:
            vgllm_items.append(("vgllm", item, f"{dataset_name}|{data_root}"))

    print(f"  [VGLLM] 共加载 {len(vgllm_items)} 条 annotation")

    # ═══════════════════════════════════════════════════════════════════
    #  第二步：合并并全局 shuffle
    # ═══════════════════════════════════════════════════════════════════
    mixed_items = mindcube_items + vgllm_items

    # 可选：限制总样本数
    if max_samples is not None and len(mixed_items) > max_samples:
        rng.shuffle(mixed_items)
        mixed_items = mixed_items[:max_samples]

    rng.shuffle(mixed_items)

    mc_count = sum(1 for src, _, _ in mixed_items if src == "mindcube")
    vg_count = sum(1 for src, _, _ in mixed_items if src == "vgllm")
    total = len(mixed_items)
    print(f"\n[Mixed] 混合结果: MindCube={mc_count}, VGLLM={vg_count}, 总计={total}")
    print(f"  实际比例: MindCube={mc_count / max(total, 1):.2%}, "
          f"VGLLM={vg_count / max(total, 1):.2%}")
    print(f"[Mixed] 全局 shuffle 完成，共 {total} 条")

    # ═══════════════════════════════════════════════════════════════════
    #  第四步：转换为 row 并写入 Parquet
    # ═══════════════════════════════════════════════════════════════════
    os.makedirs(output_dir, exist_ok=True)
    info_tracker = ParquetInfoTracker(output_dir)
    writer = ParquetBufferedWriter(
        output_dir, "mixed_mc_vgllm", ARROW_SCHEMA,
        rows_per_file, rows_per_row_group,
        info_tracker=info_tracker,
    )
    img_stats = ImageStatsCollector("Mixed")

    total_processed = 0
    total_skipped = 0

    def _convert_one_item(source_type, data_item, extra_info):
        """根据来源类型调用对应的 _item_to_row 函数。"""
        parts = extra_info.split("|", 1)
        name_or_ds = parts[0]
        root = parts[1] if len(parts) > 1 else ""

        if source_type == "mindcube":
            return _mindcube_item_to_row(data_item, root, name_or_ds, index=0)
        elif source_type == "vgllm":
            return _vgllm_item_to_row(data_item, root, name_or_ds)
        return None

    # 分批并行处理
    pbar = tqdm(total=len(mixed_items), desc="[Mixed] 写入 Parquet")
    for i in range(0, len(mixed_items), chunk_size):
        chunk = mixed_items[i:i + chunk_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_convert_one_item, src, item, info)
                for src, item, info in chunk
            ]
            results = [fut.result() for fut in futures]

        for (src, item, info), row in zip(chunk, results):
            if row is None:
                total_skipped += 1
                continue
            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "source": src,
                "id": item.get("id", item.get("qid", "")),
                "question": row["question"][:200] if row["question"] else "",
            })
            writer.add(row)
            total_processed += 1

        pbar.update(len(chunk))

    pbar.close()
    writer.finish()
    info_tracker.summary()

    print(f"\n[Mixed] 处理完成: processed={total_processed}, skipped={total_skipped}")
    img_stats.print_summary()


def process_vgllm_series(
    json_data_dir_pairs: List[Tuple[str, str]],
    output_dir: str,
    split_by_dataset: bool,
    rows_per_file: int,
    rows_per_row_group: int,
    num_workers: int = 16,
    chunk_size: int = 1024,
):
    """
    加载 VG-LLM-Data 系列数据集并转换为 Parquet 文件。

    每个 JSON/JSONL 文件对应一个子数据集，分别写入独立的 Parquet 目录。
    每个子数据集可以有自己独立的图片数据根目录。

    使用 ThreadPoolExecutor 并行处理（I/O 密集型：读取图片字节），
    JSONL 大文件采用分批（chunk）并行以控制内存占用。

    Args:
        json_data_dir_pairs: (json_path, data_dir) 元组列表，
                             每个元素指定一个 JSON/JSONL 文件及其对应的图片数据根目录。
        output_dir:        输出目录。
        split_by_dataset:  是否按子数据集分目录保存。
        rows_per_file:     每个 Parquet 文件的最大行数。
        rows_per_row_group: 每个 row group 的最大行数。
        num_workers:       并行线程数（默认 16）。
        chunk_size:        JSONL 分批大小（默认 1024），控制内存峰值。
    """
    print(f"\n[VGLLM] Processing {len(json_data_dir_pairs)} files "
          f"(num_workers={num_workers}, chunk_size={chunk_size})")
    for jp, dd in json_data_dir_pairs:
        print(f"  {jp}  →  data_dir={dd}")

    def _write_results(results, data_items, writer, img_stats, dataset_name):
        """将并行处理的结果顺序写入 writer，返回 (processed, skipped) 计数。"""
        processed = 0
        skipped = 0
        for row, data_item in zip(results, data_items):
            if row is None:
                skipped += 1
                item_id = data_item.get("id", data_item.get("qid", "unknown"))
                print(f"    [SKIP] {dataset_name} item_id={item_id} → row is None")
                continue
            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "id": data_item.get("id", data_item.get("qid", "")),
                "question": row["question"][:200],
            })
            writer.add(row)
            processed += 1
        return processed, skipped

    for json_path, data_root in json_data_dir_pairs:
        dataset_name = _infer_vgllm_dataset_name(json_path)
        data_root = data_root.replace('<', '').replace('>', '')
        print(f"\n{'─'*60}")
        print(f"  [VGLLM] {dataset_name}: {json_path}")
        print(f"{'─'*60}")

        # 为每个子数据集创建独立的 writer
        out = (os.path.join(output_dir, dataset_name)
               if split_by_dataset else output_dir)
        info_tracker = ParquetInfoTracker(out)
        writer = ParquetBufferedWriter(
            out, dataset_name, ARROW_SCHEMA,
            rows_per_file, rows_per_row_group,
            info_tracker=info_tracker,
        )

        img_stats = ImageStatsCollector(dataset_name)
        total_processed = 0
        total_skipped = 0

        is_jsonl = json_path.endswith(".jsonl")

        if is_jsonl:
            # ── JSONL 格式：分批并行处理（适合大文件如 spar_7m.jsonl） ──
            with open(json_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            print(f"  Total lines: {total_lines}")

            with open(json_path, "r", encoding="utf-8") as f:
                chunk_items: List[dict] = []   # 当前批次的数据项
                pbar = tqdm(total=total_lines, desc=f"  {dataset_name}")

                for line in f:
                    pbar.update(1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data_item = json.loads(line)
                    except Exception as e:
                        print(f"    [WARN] bad JSON line: {e}")
                        total_skipped += 1
                        continue
                    chunk_items.append(data_item)

                    # 攒满一批后并行处理
                    if len(chunk_items) >= chunk_size:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = [
                                executor.submit(_vgllm_item_to_row, item, data_root, dataset_name)
                                for item in chunk_items
                            ]
                            results = [fut.result() for fut in futures]
                        p, s = _write_results(results, chunk_items, writer, img_stats, dataset_name)
                        total_processed += p
                        total_skipped += s
                        chunk_items = []

                # 处理最后不足一批的剩余数据
                if chunk_items:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = [
                            executor.submit(_vgllm_item_to_row, item, data_root, dataset_name)
                            for item in chunk_items
                        ]
                        results = [fut.result() for fut in futures]
                    p, s = _write_results(results, chunk_items, writer, img_stats, dataset_name)
                    total_processed += p
                    total_skipped += s

                pbar.close()
        else:
            # ── JSON 数组格式：分批并行处理 ──
            print(f"  Loading JSON array from {json_path} ...")
            with open(json_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            print(f"  Total items: {len(data_list)}")

            pbar = tqdm(total=len(data_list), desc=f"  {dataset_name}")
            for i in range(0, len(data_list), chunk_size):
                chunk_items = data_list[i:i + chunk_size]
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(_vgllm_item_to_row, item, data_root, dataset_name)
                        for item in chunk_items
                    ]
                    results = [fut.result() for fut in futures]
                p, s = _write_results(results, chunk_items, writer, img_stats, dataset_name)
                total_processed += p
                total_skipped += s
                pbar.update(len(chunk_items))
            pbar.close()

        writer.finish()
        info_tracker.summary()
        print(f"  [{dataset_name}] processed={total_processed}, skipped={total_skipped}")
        img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  ScanQA – ScanNet 3D scene QA (with depth, poses, intrinsics)
# ═════════════════════════════════════════════════════════════════════════

def _load_embodiedscan_scenes(
    annotation_dir: str,
    splits: List[str] = ("train", "val", "test"),
) -> Dict[str, dict]:
    """
    Load EmbodiedScan pickle files and build a dict mapping
    ``video_id`` (e.g. ``"scannet/scene0000_00"``) → scene metadata.

    Each scene metadata dict contains at least:
      - ``images``: list of dicts with ``img_path``
      - ``axis_align_matrix``: 4×4 numpy array
      - ``depth_cam2img``: 4×4 depth intrinsic matrix
    """
    scenes: Dict[str, dict] = {}
    for split in splits:
        pkl_path = os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [WARN] EmbodiedScan pkl not found: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            data_list = pickle.load(f)["data_list"]
        for item in data_list:
            sample_idx = item.get("sample_idx", "")
            if sample_idx.startswith("scannet"):
                scenes[sample_idx] = item
    print(f"  Loaded {len(scenes)} ScanNet scenes from EmbodiedScan")
    return scenes


def _scanqa_item_to_row(
    qa_item: dict,
    scene_meta: dict,
    video_folder: str,
    video_id: str,
) -> Optional[dict]:
    """
    Convert a single ScanQA *llava_style* annotation item + scene metadata
    into the target Parquet row schema.

    The llava_style format (produced by ``process_scanqa.py``) looks like::

        {
          "id": "scanqa_val-scene0011-0_0",
          "video": "scannet/scene0011_00",
          "conversations": [
            {"value": "<image> What color ...? Answer the question simply.", "from": "human"},
            {"value": "dark brown", "from": "gpt"}
          ],
          "metadata": {
            "dataset": "scanQA",
            "question_type": "unknow",
            "answers": ["dark brown", "brown"]   // val only
          }
        }

    We store **all** raw frames of the scene (no sampling / resizing),
    with the actual image and depth bytes embedded directly in the row.

    Args:
        qa_item:      One entry from ``scanqa_{split}_llava_style.json``.
        scene_meta:   The EmbodiedScan metadata dict for this scene.
        video_folder: Root directory for ScanNet posed images
                      (the parent of ``scene0000_00/`` directories).
        video_id:     e.g. ``"scannet/scene0000_00"``

    Returns:
        A dict matching ``SCANQA_ARROW_SCHEMA``, or ``None`` on error.
    """
    try:
        # ── Collect all frame paths from the scene metadata ──
        images_meta = scene_meta["images"]
        if not images_meta:
            print(f"    [ERROR] scene_meta['images'] is empty for {video_id}")
            return None

        # img_path in embodiedscan is like "scannet/scene0000_00/00000.jpg"
        # We strip the leading "scannet/" and join with video_folder.
        image_path_list = []
        image_bytes_list = []
        depth_bytes_list = []
        for img_info in images_meta:
            img_rel = img_info["img_path"]
            # Strip leading "scannet/" so path becomes "scene0000_00/00000.jpg"
            img_rel_stripped = img_rel.replace("scannet/", "", 1)
            abs_img = os.path.join(video_folder, img_rel_stripped)
            abs_depth = abs_img.replace(".jpg", ".png")

            image_path_list.append(abs_img)

            # Read RGB image bytes
            try:
                with open(abs_img, "rb") as f:
                    image_bytes_list.append(f.read())
            except Exception as e:
                print(f"    [ERROR] cannot read image {abs_img}: {e}")
                return None

            # Read depth image bytes
            try:
                with open(abs_depth, "rb") as f:
                    depth_bytes_list.append(f.read())
            except Exception as e:
                print(f"    [ERROR] cannot read depth {abs_depth}: {e}")
                return None

        n_frames = len(image_path_list)

        # ── Per-frame poses (from .txt files next to each .jpg) ──
        # Each pose file is 4×4 cam2world. We also apply axis_align_matrix.
        axis_align = np.array(scene_meta["axis_align_matrix"]).reshape(4, 4)
        poses = []
        for img_path in image_path_list:
            pose_path = img_path.replace(".jpg", ".txt")
            try:
                raw_pose = np.loadtxt(pose_path)  # 4×4
                aligned_pose = axis_align @ raw_pose
                poses.append(aligned_pose.flatten().tolist())
            except Exception as e:
                print(f"    [ERROR] cannot read pose {pose_path}: {e}")
                return None

        # ── Depth intrinsic (shared across all frames in the scene) ──
        depth_intrinsic_raw = np.array(scene_meta["depth_cam2img"]).reshape(4, 4)
        depth_intrinsic = depth_intrinsic_raw.flatten().tolist()

        # ── Extract QA fields from llava_style conversations ──
        convs = qa_item["conversations"]
        question = ""
        answer = ""
        for turn in convs:
            role = turn["from"]
            value = turn["value"]
            if role == "human":
                question = value
            elif role == "gpt":
                answer = value

        qa_metadata = qa_item["metadata"]
        answers = qa_metadata["answers"] if "answers" in qa_metadata else [answer] if answer else []

        # ── Serialise ALL scene_meta fields into metadata ──
        # Convert numpy arrays to lists so they are JSON-serialisable.
        def _to_serialisable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: _to_serialisable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serialisable(v) for v in obj]
            return obj

        meta = {
            "type": "scanqa",
            "video_id": video_id,
            # All qa_item-level fields
            "qa_id": qa_item["id"],
            "qa_metadata": _to_serialisable(qa_metadata),
            "answers": answers,
            # ALL scene_meta fields (images, instances, cam2img,
            # axis_align_matrix, depth_cam2img, sample_idx, etc.)
            "scene_meta": _to_serialisable(scene_meta),
        }

        return {
            "question":        question,
            "answer":          answer,
            "scene_name":      "scannet",
            "dataset_name":    "scanqa",
            "image_path":      image_path_list,
            "image_bytes":     image_bytes_list,
            "depth_bytes":     depth_bytes_list,
            "poses":           poses,
            "intrinsic":       _IDENTITY_4x4,       # RGB intrinsic not in embodiedscan
            "depth_intrinsic": depth_intrinsic,
            "metadata":        _ensure_metadata_str(meta),
        }

    except Exception as e:
        print(f"    [ERROR] _scanqa_item_to_row failed for video_id={video_id}, "
              f"qa_id={qa_item.get('id', '?')}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_scanqa(
    scanqa_json_files: List[str],
    embodiedscan_dir: str,
    video_folder: str,
    writer: ParquetBufferedWriter,
    num_workers: int = 16,
):
    """
    Load pre-processed ScanQA llava_style JSON files and EmbodiedScan scene
    metadata, then convert each QA sample into a Parquet row containing:

    - All original RGB frame bytes embedded directly  (no sampling)
    - All original depth frame bytes embedded directly (no sampling)
    - Per-frame 4×4 extrinsic poses (axis-aligned)
    - Scene-level depth intrinsic matrix
    - Question / answer text + rich metadata

    No intermediate processing (frame sampling, 3D coordinate computation,
    resizing) is performed – raw image/depth bytes and annotations only.

    Uses ``SCANQA_ARROW_SCHEMA`` which stores ``image_bytes`` and
    ``depth_bytes`` (list of binary) instead of path-only columns.

    Processing is parallelised with a thread pool (I/O-bound: reading
    image/depth/pose files from disk).

    Args:
        scanqa_json_files: List of paths to ``scanqa_{split}_llava_style.json``
                           files (e.g. produced by ``process_scanqa.py``).
        embodiedscan_dir:  Directory containing
                           ``embodiedscan_infos_{train,val,test}.pkl``.
        video_folder:      Root directory for ScanNet posed images, i.e. the
                           parent of ``scene0000_00/`` directories containing
                           ``*.jpg``, ``*.png``, ``*.txt``.
        writer:            ``ParquetBufferedWriter`` to flush rows into.
        num_workers:       Number of threads for parallel I/O (default: 16).
    """
    # ── Step 1: Load all scene metadata from EmbodiedScan ──
    print(f"\n[ScanQA] Loading EmbodiedScan scene metadata from {embodiedscan_dir}")
    scene_db = _load_embodiedscan_scenes(embodiedscan_dir)

    # ── Step 2: Iterate over each provided JSON file ──
    total_processed = 0
    total_skipped = 0
    img_stats = ImageStatsCollector("ScanQA")

    for qa_path in scanqa_json_files:
        if not os.path.exists(qa_path):
            print(f"  [WARN] ScanQA file not found: {qa_path}")
            continue

        label = os.path.basename(qa_path)
        print(f"\n  [ScanQA/{label}] Loading {qa_path}")
        with open(qa_path, "r") as f:
            qa_items = json.load(f)
        print(f"  [ScanQA/{label}] {len(qa_items)} QA items")

        # ── Filter items that have a valid scene in scene_db ──
        valid_tasks = []
        pre_skipped = 0
        for qa_item in qa_items:
            video_id = qa_item["video"]
            if not video_id:
                pre_skipped += 1
                continue
            if video_id not in scene_db:
                pre_skipped += 1
                continue
            valid_tasks.append((qa_item, scene_db[video_id], video_folder, video_id))

        print(f"  [ScanQA/{label}] {len(valid_tasks)} valid tasks, "
              f"{pre_skipped} skipped (no scene meta)")

        # ── Parallel processing with ThreadPoolExecutor ──
        skipped = pre_skipped
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, (qa_item, scene_meta, vf, vid) in enumerate(valid_tasks):
                future = executor.submit(
                    _scanqa_item_to_row, qa_item, scene_meta, vf, vid,
                )
                future_to_idx[future] = idx

            # Collect results in submission order for deterministic output
            results = [None] * len(valid_tasks)
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(future_to_idx),
                desc=f"  ScanQA/{label}",
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"    [ERROR] Future {idx} raised: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    results[idx] = None

        # ── Write results in order ──
        for idx, row in enumerate(results):
            if row is None:
                skipped += 1
                continue

            qa_item, _, _, video_id = valid_tasks[idx]
            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "video_id": video_id,
                "id": qa_item["id"],
                "n_frames": n_images,
                "question": row["question"][:200],
            })

            writer.add(row)

        n = len(qa_items)
        total_processed += n - skipped
        total_skipped += skipped
        print(f"  [ScanQA/{label}] processed={n - skipped}, skipped={skipped}")

    print(f"[ScanQA] total processed={total_processed}, total skipped={total_skipped}")
    img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  Video3DLLM – VEGA-3D 五个数据集（ScanQA/SQA3D/Scan2Cap/ScanRefer/Multi3DRefer）
#  采用 VEGA-3D 的 task-grouped shuffle 策略排布数据
#  离线预计算：unproject 世界坐标、场景边界 boundary、物体框 objects
# ═════════════════════════════════════════════════════════════════════════

# VEGA-3D 任务分组（与 llava_trainer.py 中的 task_mapping 一致）
_VIDEO3DLLM_TASK_GROUPS = {
    "scanqa":       0,   # QA 任务
    "sqa3d":        0,   # QA 任务
    "scan2cap":     1,   # Caption 任务
    "scanrefer":    2,   # Grounding 任务
    "multi3drefer": 2,   # Grounding 任务
}

# 从 JSON 文件名推断子数据集名称
_VIDEO3DLLM_FILENAME_TO_DATASET = {
    "scanqa":       "scanqa",
    "sqa3d":        "sqa3d",
    "scan2cap":     "scan2cap",
    "scanrefer":    "scanrefer",
    "multi3drefer": "multi3drefer",
}


def _infer_video3dllm_dataset_name(filepath: str) -> str:
    """根据文件名推断 Video3DLLM 子数据集名称。"""
    basename = os.path.basename(filepath).lower()
    for key, ds_name in _VIDEO3DLLM_FILENAME_TO_DATASET.items():
        if key in basename:
            return ds_name
    stem = os.path.splitext(basename)[0]
    return stem


def _unproject_numpy(intrinsics: np.ndarray, poses: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    深度图反投影为世界坐标（纯 NumPy 实现，无需 PyTorch）。

    Args:
        intrinsics: (V, 4, 4) float32，相机内参矩阵
        poses:      (V, 4, 4) float32，cam2world 外参矩阵（已应用 axis_align）
        depths:     (V, H, W) float32，深度图（单位：毫米，除以 1000 得到米）

    Returns:
        world_coords: (V, H, W, 3) float32，世界坐标
    """
    V, H, W = depths.shape
    # 像素坐标网格
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    world_coords = np.zeros((V, H, W, 3), dtype=np.float32)

    for i in range(V):
        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]

        z = depths[i] / 1000.0  # (H, W)，毫米 → 米
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        # 相机坐标 (H*W, 4)
        ones = np.ones((H * W,), dtype=np.float32)
        cam_coords = np.stack([
            x.reshape(-1), y.reshape(-1), z.reshape(-1), ones
        ], axis=1)  # (H*W, 4)

        # 变换到世界坐标
        wc = (poses[i] @ cam_coords.T).T  # (H*W, 4)
        wc = wc[:, :3] / wc[:, 3:4]       # 齐次除法

        world_coords[i] = wc.reshape(H, W, 3)

    return world_coords


def _video3dllm_item_to_row(
    item: dict,
    scene_meta: dict,
    video_folder: str,
    video_id: str,
    scan2obj: dict,
    dataset_name: str,
) -> Optional[dict]:
    """
    将一条 VEGA-3D 格式的标注转换为 VIDEO3DLLM_ARROW_SCHEMA 行。

    VEGA-3D llava_style JSON 格式::

        {
          "id": "...",
          "video": "scannet/scene0415_00",
          "conversations": [
            {"from": "human", "value": "<image>..."},
            {"from": "gpt",   "value": "..."}
          ],
          "metadata": {"dataset": "scanrefer", ...},
          "box_input": [x, y, z],          # scan2cap 专用
          "box": [[x1,y1,z1,...], ...],    # scanrefer/multi3drefer 专用
        }

    离线预计算：
      - 所有帧的图片路径和深度图路径
      - 每帧 cam2world 位姿（已应用 axis_align_matrix）
      - unproject 世界坐标 (V, H, W, 3)
      - 场景边界 boundary [x_min, x_max, y_min, y_max, z_min, z_max]
      - 物体框 objects（来自 scan2obj）

    Args:
        item:         单条 VEGA-3D 标注。
        scene_meta:   EmbodiedScan 场景元数据。
        video_folder: ScanNet posed images 根目录。
        video_id:     e.g. "scannet/scene0415_00"
        scan2obj:     scene_id → objects 列表的映射（来自 scannet_{split}_gt_box.json）
        dataset_name: 子数据集名称（"scanqa"/"sqa3d"/"scan2cap"/"scanrefer"/"multi3drefer"）

    Returns:
        符合 VIDEO3DLLM_ARROW_SCHEMA 的 dict，或 None（出错时）。
    """
    try:
        # ── Step 1: 收集所有帧路径 ──
        images_meta = scene_meta["images"]
        if not images_meta:
            print(f"    [ERROR] scene_meta['images'] is empty for {video_id}")
            return None

        image_path_list = []
        depth_path_list = []
        for img_info in images_meta:
            img_rel = img_info["img_path"]
            # EmbodiedScan 中路径格式: "scannet/scene0000_00/00000.jpg"
            # 去掉前缀 "scannet/" 后拼接 video_folder
            img_rel_stripped = img_rel.replace("scannet/", "", 1)
            abs_img = os.path.join(video_folder, img_rel_stripped)
            abs_depth = abs_img.replace(".jpg", ".png")

            image_path_list.append(abs_img)
            depth_path_list.append(abs_depth)

        n_frames = len(image_path_list)

        # ── Step 2: 读取每帧位姿（cam2world，已应用 axis_align） ──
        axis_align = np.array(scene_meta["axis_align_matrix"]).reshape(4, 4).astype(np.float32)
        poses_list = []
        for img_path in image_path_list:
            pose_path = img_path.replace(".jpg", ".txt")
            try:
                raw_pose = np.loadtxt(pose_path).astype(np.float32)  # (4, 4)
                aligned_pose = (axis_align @ raw_pose).astype(np.float32)
                poses_list.append(aligned_pose.flatten().tolist())
            except Exception as e:
                print(f"    [ERROR] cannot read pose {pose_path}: {e}")
                return None

        # ── Step 3: 深度内参（场景共享） ──
        depth_intrinsic_raw = np.array(scene_meta["depth_cam2img"]).reshape(4, 4).astype(np.float32)
        depth_intrinsic = depth_intrinsic_raw.flatten().tolist()

        # ── Step 4: 读取深度图并计算世界坐标 ──
        depths = []
        for depth_path in depth_path_list:
            try:
                with Image.open(depth_path) as depth_img:
                    d = np.array(depth_img).astype(np.float32)  # (H, W)，单位毫米
                depths.append(d)
            except Exception as e:
                print(f"    [ERROR] cannot read depth {depth_path}: {e}")
                return None

        depths_arr = np.stack(depths, axis=0)  # (V, H, W)
        poses_arr = np.array([np.array(p).reshape(4, 4) for p in poses_list], dtype=np.float32)  # (V, 4, 4)
        intrinsics_arr = np.tile(depth_intrinsic_raw, (n_frames, 1, 1))  # (V, 4, 4)

        # unproject：深度图反投影为世界坐标
        world_coords = _unproject_numpy(intrinsics_arr, poses_arr, depths_arr)  # (V, H, W, 3)

        # ── Step 5: 计算场景边界 boundary ──
        wc_flat = world_coords.reshape(-1, 3)
        # 过滤掉深度为 0 的无效点（深度=0 → z=0 → 世界坐标可能无效）
        valid_mask = depths_arr.reshape(-1) > 0
        if valid_mask.sum() > 0:
            wc_valid = wc_flat[valid_mask]
            x_min, x_max = float(wc_valid[:, 0].min()), float(wc_valid[:, 0].max())
            y_min, y_max = float(wc_valid[:, 1].min()), float(wc_valid[:, 1].max())
            z_min, z_max = float(wc_valid[:, 2].min()), float(wc_valid[:, 2].max())
        else:
            x_min = x_max = y_min = y_max = z_min = z_max = 0.0
        boundary = [x_min, x_max, y_min, y_max, z_min, z_max]

        # ── Step 6: 加载物体框 objects ──
        # scan2obj 的 key 是 scene_id（不含 "scannet/" 前缀）
        scene_id = video_id.split("/")[-1]  # e.g. "scene0415_00"
        objects = scan2obj.get(video_id, scan2obj.get(scene_id, []))
        objects_json = json.dumps(objects, ensure_ascii=False)

        # ── Step 7: 序列化世界坐标（降采样以控制存储大小） ──
        # 将 (V, H, W, 3) float32 序列化为 JSON 字符串
        # 注意：原始分辨率可能很大（如 480×640），这里存储完整数据
        # und_dataset 在线处理时会根据帧采样索引取子集
        world_coords_json = json.dumps(world_coords.tolist(), ensure_ascii=False)

        # ── Step 8: 提取 question / answer ──
        convs = item.get("conversations", [])
        question = ""
        answer = ""
        for turn in convs:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                question = value
            elif role == "gpt":
                answer = value

        # ── Step 9: 构建 metadata ──
        item_meta = item.get("metadata", {})
        meta = {
            "type": f"video3dllm_{dataset_name}",
            "id": item.get("id", ""),
            "video_id": video_id,
            "dataset": dataset_name,
            "original_metadata": item_meta,
        }
        # 保留各子数据集的特殊字段
        if dataset_name == "scan2cap" and "box_input" in item:
            meta["box_input"] = item["box_input"]
        if dataset_name in ("scanrefer", "multi3drefer") and "box" in item:
            meta["box"] = item["box"]

        return {
            "question":        question,
            "answer":          answer,
            "scene_name":      "scannet",
            "dataset_name":    f"video3dllm_{dataset_name}",
            "image_path":      image_path_list,
            "depth_list":      depth_path_list,
            "poses":           poses_list,
            "intrinsic":       _IDENTITY_4x4,
            "depth_intrinsic": depth_intrinsic,
            "metadata":        _ensure_metadata_str(meta),
            "world_coords":    world_coords_json,
            "boundary":        boundary,
            "objects":         objects_json,
        }

    except Exception as e:
        print(f"    [ERROR] _video3dllm_item_to_row failed for video_id={video_id}, "
              f"id={item.get('id', '?')}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_video3dllm(
    json_data_dir_pairs: List[Tuple[str, str]],
    embodiedscan_dir: str,
    video_folder: str,
    scan2obj_files: List[str],
    output_dir: str,
    rows_per_file: int,
    rows_per_row_group: int,
    shuffle_seed: int = 42,
    num_workers: int = 16,
    chunk_size: int = 256,
    max_samples: Optional[int] = None,
):
    """
    处理 VEGA-3D 五个数据集（ScanQA/SQA3D/Scan2Cap/ScanRefer/Multi3DRefer），
    采用 VEGA-3D 的 task-grouped shuffle 策略排布数据，
    离线预计算 unproject 世界坐标、场景边界 boundary、物体框 objects。

    Shuffle 策略（与 VEGA-3D llava_trainer.py 中的 task-grouped shuffle 一致）：
      1. 按任务类型分组：QA组(scanqa+sqa3d)、Caption组(scan2cap)、Grounding组(scanrefer+multi3drefer)
      2. 每组内部独立 shuffle
      3. 每组切分为 megabatch（大小 = megabatch_size）
      4. 所有组的 megabatch 合并后随机 shuffle（保证 batch 内任务一致性）

    Args:
        json_data_dir_pairs:  [(json_path, data_root), ...] 每个数据集的 JSON 文件和数据根目录
        embodiedscan_dir:     EmbodiedScan pkl 文件目录
        video_folder:         ScanNet posed images 根目录（scene0000_00/ 的父目录）
        scan2obj_files:       scannet_{split}_gt_box.json 文件列表
        output_dir:           输出 Parquet 目录
        rows_per_file:        每个 Parquet 文件的最大行数
        rows_per_row_group:   每个 row group 的最大行数
        shuffle_seed:         随机种子
        num_workers:          并行线程数（I/O 密集型）
        chunk_size:           并行处理的批大小
        max_samples:          最大总样本数（None 表示不限制）
    """
    import random as _random
    rng = _random.Random(shuffle_seed)

    # ── Step 1: 加载 EmbodiedScan 场景元数据 ──
    print(f"\n[Video3DLLM] 加载 EmbodiedScan 场景元数据: {embodiedscan_dir}")
    scene_db = _load_embodiedscan_scenes(embodiedscan_dir)

    # ── Step 2: 加载物体框 scan2obj ──
    scan2obj: Dict[str, list] = {}
    for box_file in scan2obj_files:
        if not os.path.exists(box_file):
            print(f"  [WARN] scan2obj file not found: {box_file}")
            continue
        with open(box_file, "r") as f:
            data = json.load(f)
        scan2obj.update(data)
    print(f"  [Video3DLLM] 加载 {len(scan2obj)} 个场景的物体框")

    # ── Step 3: 加载所有数据集的标注，按任务分组 ──
    # task_groups: {task_id: [(item, scene_meta, video_folder, video_id, dataset_name), ...]}
    task_groups: Dict[int, list] = {0: [], 1: [], 2: []}

    for json_path, data_root in json_data_dir_pairs:
        dataset_name = _infer_video3dllm_dataset_name(json_path)
        task_id = _VIDEO3DLLM_TASK_GROUPS.get(dataset_name, 0)

        if not os.path.exists(json_path):
            print(f"  [WARN] JSON file not found: {json_path}")
            continue

        print(f"\n  [Video3DLLM/{dataset_name}] 加载 {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        print(f"  [Video3DLLM/{dataset_name}] {len(items)} 条标注，task_id={task_id}")

        # 过滤有效场景
        valid_count = 0
        skip_count = 0
        for item in items:
            video_id = item.get("video", "")
            if not video_id:
                skip_count += 1
                continue
            if video_id not in scene_db:
                skip_count += 1
                continue
            task_groups[task_id].append(
                (item, scene_db[video_id], video_folder, video_id, dataset_name)
            )
            valid_count += 1

        print(f"  [Video3DLLM/{dataset_name}] valid={valid_count}, skipped={skip_count}")

    for tid, group in task_groups.items():
        print(f"  [Video3DLLM] task_group={tid}: {len(group)} 条")

    # ── Step 4: VEGA-3D task-grouped shuffle ──
    # 策略：每组内部 shuffle → 切分 megabatch → 跨组 megabatch 随机 shuffle
    megabatch_size = 256  # 与 VEGA-3D 训练时的 world_size * batch_size 对齐

    all_megabatches = []
    for task_id in sorted(task_groups.keys()):
        group = task_groups[task_id]
        if not group:
            continue
        # 组内 shuffle
        rng.shuffle(group)
        # 切分为 megabatch（丢弃最后一个不完整的 megabatch）
        for start in range(0, len(group) - megabatch_size + 1, megabatch_size):
            all_megabatches.append(group[start:start + megabatch_size])
        # 保留最后一个不完整的 megabatch（与 VEGA-3D 不同，这里不丢弃）
        remainder = len(group) % megabatch_size
        if remainder > 0:
            all_megabatches.append(group[len(group) - remainder:])

    # megabatch 间随机 shuffle（保证不同任务类型交错出现）
    rng.shuffle(all_megabatches)

    # 展开为最终的有序列表
    ordered_items = []
    for mb in all_megabatches:
        ordered_items.extend(mb)

    total = len(ordered_items)
    print(f"\n[Video3DLLM] task-grouped shuffle 完成，共 {total} 条")

    # 可选：限制总样本数
    if max_samples is not None and total > max_samples:
        ordered_items = ordered_items[:max_samples]
        total = len(ordered_items)
        print(f"  [Video3DLLM] 截断至 {total} 条（max_samples={max_samples}）")

    # ── Step 5: 并行转换并写入 Parquet ──
    os.makedirs(output_dir, exist_ok=True)
    info_tracker = ParquetInfoTracker(output_dir)
    writer = ParquetBufferedWriter(
        output_dir, "video3dllm", VIDEO3DLLM_ARROW_SCHEMA,
        rows_per_file, rows_per_row_group,
        info_tracker=info_tracker,
    )
    img_stats = ImageStatsCollector("Video3DLLM")

    total_processed = 0
    total_skipped = 0

    def _convert_one(args_tuple):
        item, scene_meta, vf, vid, ds_name = args_tuple
        return _video3dllm_item_to_row(item, scene_meta, vf, vid, scan2obj, ds_name)

    pbar = tqdm(total=total, desc="[Video3DLLM] 写入 Parquet")
    for i in range(0, total, chunk_size):
        chunk = ordered_items[i:i + chunk_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_convert_one, args_tuple): j
                for j, args_tuple in enumerate(chunk)
            }
            results = [None] * len(chunk)
            for future in concurrent.futures.as_completed(future_to_idx):
                j = future_to_idx[future]
                try:
                    results[j] = future.result()
                except Exception as e:
                    print(f"    [ERROR] chunk[{j}] raised: {type(e).__name__}: {e}")
                    results[j] = None

        for j, row in enumerate(results):
            if row is None:
                total_skipped += 1
                continue
            item, _, _, video_id, ds_name = chunk[j]
            n_images = len(row["image_path"])
            img_stats.record(n_images, {
                "video_id": video_id,
                "dataset": ds_name,
                "n_frames": n_images,
                "question": row["question"][:100],
            })
            writer.add(row)
            total_processed += 1

        pbar.update(len(chunk))

    pbar.close()
    writer.finish()
    info_tracker.summary()
    print(f"[Video3DLLM] total processed={total_processed}, total skipped={total_skipped}")
    img_stats.print_summary()


# ═════════════════════════════════════════════════════════════════════════
#  Ego3D-Bench → TSV (for VLMEvalKit)
# ═════════════════════════════════════════════════════════════════════════

# 不同数据源的图像视角排序（与原始 Ego3D-Bench 仓库一致）
_EGO3D_IMAGE_ORDER = {
    'nuscenes':  ['Front_Left', 'Front', 'Front_Right', 'Back_Right', 'Back', 'Back_Left'],
    'waymo':     ['Front', 'Front_Left', 'Side_Left', 'Front_Right', 'Side_Right'],
    'argoverse': ['Front_Left', 'Front', 'Front_Right', 'Side_Right', 'Back_Right', 'Back_Left', 'Side_Left'],
}


def process_ego3d_to_tsv(
    data_root: str,
    tsv_output_path: str,
    hf_repo: Optional[str] = None,
):
    """
    将本地下载的 Ego3D-Bench 数据（HuggingFace Arrow 格式）转换为
    VLMEvalKit 可加载的 TSV 文件，并可选上传到 HuggingFace。

    数据目录结构::

        data_root/
        ├── test/
        │   └── data-00000-of-00001.arrow
        └── raw_images/
            └── *.jpg

    原始数据每条记录包含:
      - question:  str  (含 <image> 占位符)
      - answer:    str
      - options:   list[str] | None
      - category:  str  (10 个子类别)
      - source:    str  (nuscenes / waymo / argoverse)
      - images:    dict  {视角名: 相对路径}

    输出 TSV 列:
      index, question, answer, options, category, source, image_path
    """
    import pyarrow.ipc as ipc

    # 直接读取 Arrow 文件（目录下可能没有 HuggingFace Dataset 元数据）
    arrow_path = os.path.join(data_root, 'test', 'data-00000-of-00001.arrow')
    if not os.path.isfile(arrow_path):
        raise FileNotFoundError(
            f"Arrow file not found: {arrow_path}\n"
            f"请确认 data_root 指向包含 test/ 子目录的 Ego3D-Bench 根目录。"
        )

    print(f"\n[Ego3D-Bench] Loading Arrow file: {arrow_path}")
    with open(arrow_path, 'rb') as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    dataset = table.to_pydict()          # dict[str, list]
    num_samples = len(next(iter(dataset.values())))

    print(f"  Loaded {num_samples} samples")
    print(f"  Columns: {list(dataset.keys())}")

    raw_images_dir = os.path.join(data_root, 'raw_images')
    if not os.path.isdir(raw_images_dir):
        # 尝试上一级目录
        raw_images_dir_alt = os.path.join(os.path.dirname(data_root), 'raw_images')
        if os.path.isdir(raw_images_dir_alt):
            raw_images_dir = raw_images_dir_alt
        else:
            print(f"  [WARN] raw_images directory not found at {raw_images_dir}")
            raw_images_dir = data_root  # fallback

    rows = []
    skipped = 0

    for idx in tqdm(range(num_samples), desc="  Ego3D-Bench → TSV"):
        question = dataset.get('question', [''])[idx] if 'question' in dataset else ''
        answer = dataset.get('answer', [''])[idx] if 'answer' in dataset else ''
        options = dataset.get('options', [None])[idx] if 'options' in dataset else None
        category = dataset.get('category', [''])[idx] if 'category' in dataset else ''
        source = dataset.get('source', [''])[idx] if 'source' in dataset else ''
        images_dict = dataset.get('images', [{}])[idx] if 'images' in dataset else {}

        if not images_dict:
            skipped += 1
            continue

        # 按数据源确定图像视角排序
        image_order = _EGO3D_IMAGE_ORDER.get(source, list(images_dict.keys()))

        # 构建排序后的**相对路径**列表
        # 保存相对于 HuggingFace 仓库根目录的路径（如 raw_images/xxx.jpg），
        # ego3d.py 的 prepare_tsv 会在运行时将其拼接为绝对路径。
        image_paths = []
        all_found = True
        for view_name in image_order:
            if view_name not in images_dict:
                continue
            rel_path = images_dict[view_name]

            # 验证图片文件存在（用绝对路径检查）
            found_abs = None
            candidates = [
                os.path.join(raw_images_dir, os.path.basename(rel_path)),
                os.path.join(data_root, rel_path),
                os.path.join(raw_images_dir, rel_path),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    found_abs = cand
                    break

            if found_abs is None:
                print(f"    [WARN] Image not found for row {idx}, view={view_name}: "
                      f"tried {candidates}")
                all_found = False
                break

            # 计算相对于 data_root 的路径
            rel_to_root = os.path.relpath(found_abs, data_root)
            image_paths.append(rel_to_root)

        if not image_paths or not all_found:
            skipped += 1
            continue

        # 序列化 options 和 image_path 为 Python list 字符串（单引号）
        # 注意：不能用 json.dumps（双引号），否则与 pandas to_csv 的引号转义冲突，
        # 导致含换行符的 question 字段在 pd.read_csv 时解析失败。
        # 使用 str() 生成单引号格式，与 MindCube TSV 保持一致。
        options_str = str(options) if options else str([])
        image_path_str = str(image_paths)

        rows.append({
            'index':      idx,
            'question':   question,
            'answer':     answer,
            'options':    options_str,
            'category':   category,
            'source':     source,
            'image_path': image_path_str,
        })

    print(f"  [Ego3D-Bench] processed={len(rows)}, skipped={skipped}")

    if not rows:
        print("  [ERROR] No valid rows produced. Check data paths.")
        return

    # 统计各 category 的数量
    from collections import Counter
    cat_counts = Counter(r['category'] for r in rows)
    print("  Category distribution:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    # 保存为 TSV
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(tsv_output_path) or '.', exist_ok=True)
    df.to_csv(tsv_output_path, sep='\t', index=False, encoding='utf-8')
    print(f"\n  ✓ TSV saved to {tsv_output_path}  ({len(df)} rows)")

    # 可选上传到 HuggingFace
    if hf_repo:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("  [ERROR] 'huggingface_hub' package is required for HF upload. "
                  "pip install huggingface_hub")
            return

        print(f"  Uploading TSV to HuggingFace repo: {hf_repo} ...")
        api = HfApi()
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=tsv_output_path,
            path_in_repo=os.path.basename(tsv_output_path),
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/datasets/{hf_repo}")


# ═════════════════════════════════════════════════════════════════════════
#  CSV saving & HuggingFace upload helper
# ═════════════════════════════════════════════════════════════════════════

def _save_csv_and_upload_hf(
    rows: List[dict],
    csv_output_path: Optional[str],
    hf_repo: Optional[str],
    dataset_label: str = "dataset",
):
    """
    Save collected rows to a local CSV file and optionally upload to
    HuggingFace Hub as a dataset.

    List/dict columns are JSON-serialised so that CSV stays readable.
    """
    # ── Prepare DataFrame ──
    # Serialise non-scalar columns to JSON strings for CSV compatibility
    df_rows = []
    for r in rows:
        flat = {}
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        df_rows.append(flat)
    df = pd.DataFrame(df_rows)

    # ── Save as TSV locally (VLMEvalKit loads .tsv with tab separator) ──
    if csv_output_path is None:
        csv_output_path = f"{dataset_label}_export.tsv"
    # Ensure the extension is .tsv for compatibility with VLMEvalKit's load()
    if csv_output_path.endswith(".csv"):
        csv_output_path = csv_output_path[:-4] + ".tsv"
    os.makedirs(os.path.dirname(csv_output_path) or ".", exist_ok=True)
    df.to_csv(csv_output_path, sep='\t', index=False, encoding="utf-8")
    print(f"\n  ✓ TSV saved to {csv_output_path}  ({len(df)} rows)")

    # ── Upload to HuggingFace Hub ──
    if hf_repo:
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
        except ImportError:
            print("  [ERROR] 'datasets' and 'huggingface_hub' packages are "
                  "required for HF upload.  pip install datasets huggingface_hub")
            return

        print(f"  Uploading CSV to HuggingFace repo: {hf_repo} ...")
        api = HfApi()
        # Create the repo if it doesn't exist (dataset type)
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
        # Upload the CSV file
        api.upload_file(
            path_or_fileobj=csv_output_path,
            path_in_repo=os.path.basename(csv_output_path),
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/datasets/{hf_repo}")


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert spatial-reasoning datasets into Parquet "
                    "files for ReconthenUndIterableDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── SPAR ──
    p.add_argument(
        "--spar-json", type=str, default=None,
        help="Path to the unified SPAR JSON file "
             "(e.g. 1rxr_1scannet_1scannetpp_1structured3d_7m.json)",
    )

    # ── MindCube ──
    p.add_argument("--mindcube-json", type=str, default=None,
                   help="Path to MindCube mix JSON file "
                        "(e.g. MindCube_train_80k.json)")
    p.add_argument("--mindcube-data-root", type=str,
                   default="/data/spatial_data/reason_data",
                   help="Root directory that mindcube annotation/image "
                        "relative paths are resolved against")

    # ── OmniSpatial ──
    p.add_argument("--omnispatial-path", type=str, default=None,
                   help="Path to OmniSpatial-test directory")
    p.add_argument("--omnispatial-prompt-type", type=str, default="manual_cot",
                   choices=["direct", "manual_cot"])
    p.add_argument("--omnispatial-eval-type", type=str, default="re",
                   choices=["re", "mc"])

    # ── OST-Bench ──
    p.add_argument("--ost-file", type=str, default=None,
                   help="Path to OST-Bench JSON file")
    p.add_argument("--ost-image-root", type=str, default=None,
                   help="Image root for OST-Bench "
                        "(e.g. .../img_train/)")

    # ── SenseNova-SI ──
    p.add_argument("--sensenova-si-jsonl", type=str, default=None,
                   help="Path to SenseNova-SI-800K.jsonl file")
    p.add_argument("--sensenova-si-image-root", type=str, default=None,
                   help="Root directory containing the 'images/' folder "
                        "for SenseNova-SI (the snapshot directory)")

    # ── RE10K ──
    p.add_argument("--re10k-data-root", type=str, default=None,
                   help="Path to RE10K snapshot directory containing scene "
                        "sub-folders (e.g. .../snapshots/<hash>/)")
    p.add_argument("--re10k-min-frames", type=int, default=2,
                   help="Skip RE10K scenes with fewer frames (default: 2)")
    p.add_argument("--re10k-max-frames", type=int, default=0,
                   help="Skip RE10K scenes with more frames (0=no limit)")

    # ── ARKitScenes ──
    p.add_argument("--arkitscenes-data-root", type=str, default=None,
                   help="Path to ARKitScenes Training directory "
                        "(e.g. .../ar_raw_all/raw/Training)")
    p.add_argument("--arkitscenes-max-frames", type=int, default=0,
                   help="Max frames per ARKitScenes scene (0=no limit)")
    p.add_argument("--arkitscenes-frame-stride", type=int, default=1,
                   help="Sub-sample every N-th frame for ARKitScenes (default: 1)")
    # ── SPLD (SpatialLadder) ──
    p.add_argument("--spld-data-root", type=str, default=None,
                   help="Path to SpatialLadder-26k snapshot directory "
                        "(containing scene folders and JSONL blobs)")
    # ── VGLLM (VG-LLM-Data series) ──
    p.add_argument("--vgllm-json", type=str, nargs="+", default=None,
                   help="One or more VG-LLM entries in 'json_path::data_dir' "
                        "format, where data_dir is the root directory for "
                        "resolving image relative paths of that JSON file. "
                        "Example: '/path/to/scan2cap.json::/data/scannet_root'")
    p.add_argument("--vgllm-num-workers", type=int, default=16,
                   help="Number of threads for parallel VGLLM I/O "
                        "(default: 16)")
    p.add_argument("--vgllm-chunk-size", type=int, default=1024,
                   help="JSONL batch size for chunked parallel processing "
                        "(default: 1024). Controls memory peak.")
    # ── ScanQA ──
    p.add_argument("--scanqa-json", type=str, nargs="+", default=None,
                   help="One or more scanqa_{split}_llava_style.json files "
                        "(e.g. .../processed/scanqa_train_llava_style.json "
                        ".../processed/scanqa_val_llava_style.json)")
    p.add_argument("--scanqa-embodiedscan-dir", type=str, default=None,
                   help="Directory containing embodiedscan_infos_{train,val,test}.pkl")
    p.add_argument("--scanqa-video-folder", type=str, default=None,
                   help="Root directory for ScanNet posed images, i.e. the "
                        "parent of scene0000_00/ directories containing "
                        "*.jpg, *.png, *.txt")
    p.add_argument("--scanqa-num-workers", type=int, default=16,
                   help="Number of threads for parallel ScanQA I/O "
                        "(default: 16)")
    # ── Video3DLLM (VEGA-3D 五个数据集) ──
    p.add_argument("--video3dllm-json", type=str, nargs="+", default=None,
                   help="One or more Video3DLLM entries in 'json_path::data_root' format. "
                        "支持的数据集：scanqa/sqa3d/scan2cap/scanrefer/multi3drefer。"
                        "Example: '/path/to/scanrefer_vg_train_llava_style.json::/data/scannet'")
    p.add_argument("--video3dllm-embodiedscan-dir", type=str, default=None,
                   help="Directory containing embodiedscan_infos_{train,val,test}.pkl "
                        "(与 --scanqa-embodiedscan-dir 相同)")
    p.add_argument("--video3dllm-video-folder", type=str, default=None,
                   help="Root directory for ScanNet posed images "
                        "(与 --scanqa-video-folder 相同)")
    p.add_argument("--video3dllm-scan2obj-files", type=str, nargs="+", default=None,
                   help="scannet_{split}_gt_box.json 文件列表，用于加载物体框。"
                        "Example: '/path/to/scannet_train_gt_box.json /path/to/scannet_val_gt_box.json'")
    p.add_argument("--video3dllm-num-workers", type=int, default=8,
                   help="Number of threads for parallel Video3DLLM I/O (default: 8)")
    p.add_argument("--video3dllm-chunk-size", type=int, default=128,
                   help="Batch size for chunked parallel processing (default: 128)")
    p.add_argument("--video3dllm-shuffle-seed", type=int, default=42,
                   help="Random seed for task-grouped shuffle (default: 42)")
    p.add_argument("--video3dllm-max-samples", type=int, default=None,
                   help="Max total samples (default: no limit)")
    # ── Mixed MindCube + VGLLM ──
    p.add_argument("--mixed-mindcube-json", type=str, nargs="+", default=None,
                   help="One or more MindCube entries in 'json_path::data_dir::repeat_time' "
                        "format. repeat_time controls data repetition (>1) or sub-sampling (<1). "
                        "Example: '/path/to/MindCube_train_80k.json::/data/root::2'")
    p.add_argument("--mixed-vgllm-json", type=str, nargs="+", default=None,
                   help="One or more VG-LLM entries in 'json_path::data_dir::repeat_time' "
                        "format. repeat_time controls data repetition (>1) or sub-sampling (<1). "
                        "Example: '/path/to/spar_234k.json::/data/spar::1'")
    p.add_argument("--mixed-max-samples", type=int, default=None,
                   help="Max total samples in mixed mode (default: no limit)")
    p.add_argument("--mixed-shuffle-seed", type=int, default=42,
                   help="Random seed for mixed shuffle (default: 42)")
    p.add_argument("--mixed-num-workers", type=int, default=16,
                   help="Number of threads for mixed parallel I/O (default: 16)")
    p.add_argument("--mixed-chunk-size", type=int, default=1024,
                   help="Batch size for chunked parallel processing in mixed mode (default: 1024)")

    # ── Annotation Control ──
    p.add_argument("--need-3d-annotation", action="store_true", default=False,
        help="Whether to load 3D annotation data (poses, depth, intrinsics). "
            "If False, output directory will have 'no_3d_annotation' suffix."
    )

    # ── 3DThinker-10K ──
    p.add_argument("--thinker3d-jsonl", type=str, default=None,
                   help="Path to 3DThinker-10K JSONL file "
                        "(e.g. data_output3d_begin_10k_resized.jsonl)")
    p.add_argument("--thinker3d-data-root", type=str, default=None,
                   help="Root directory containing other_all_image_resize/ "
                        "(the 3DThinker-10K dataset directory)")

    # ── Ego3D-Bench ──
    p.add_argument("--ego3d-data-root", type=str, default=None,
                   help="Path to the local Ego3D-Bench HuggingFace dataset directory "
                        "(containing test/ and raw_images/ sub-directories). "
                        "Example: /path/to/Ego3D-Bench/Ego3D-Bench")
    p.add_argument("--ego3d-tsv-output", type=str, default=None,
                   help="Path to save the Ego3D-Bench TSV file locally "
                        "(default: <output-dir>/Ego3D-Bench.tsv)")
    p.add_argument("--ego3d-hf-repo", type=str, default=None,
                   help="HuggingFace Hub repo id to upload Ego3D-Bench TSV "
                        "(e.g. 'lmms-lab-si/EASI-Leaderboard-Data'). "
                        "Requires `huggingface-cli login` beforehand.")

    # ── CSV & HuggingFace ──
    p.add_argument("--csv-output", type=str, default=None,
                   help="Path to save the MindCube TSV file locally "
                        "(default: <output-dir>/mindcube_export.tsv)")
    p.add_argument("--hf-repo", type=str, default=None,
                   help="HuggingFace Hub repo id to upload CSV "
                        "(e.g. 'your-username/mindcube-spatial'). "
                        "Requires `huggingface-cli login` beforehand.")

    # ── Output ──
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for Parquet files")
    p.add_argument("--rows-per-row-group", type=int, default=500,
                   help="Rows per Parquet row group (default: 500)")
    p.add_argument("--rows-per-file", type=int, default=5000,
                   help="Max rows per Parquet file (default: 5000)")
    p.add_argument("--split-by-dataset", action="store_true",
                   help="Write separate sub-dirs per dataset")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Shared tracker: writes parquet_info.json after every flush
    info_tracker = ParquetInfoTracker(args.output_dir)

    any_dataset = False

    # ------------------------------------------------------------------
    #  SPAR
    # ------------------------------------------------------------------
    if args.spar_json is not None:
        any_dataset = True
        # Determine output directory based on 3D annotation requirement
        if args.need_3d_annotation:
            spar_subdir = "spar"
        else:
            spar_subdir = "spar_no_3d_annotation"
        
        out = (os.path.join(args.output_dir, spar_subdir)
           if args.split_by_dataset else args.output_dir)

        writer = ParquetBufferedWriter(
            out, "spar", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=info_tracker,
        )
        process_spar(args.spar_json, writer, need_3d_annotation=args.need_3d_annotation)
        writer.finish()

    # ------------------------------------------------------------------
    #  MindCube
    # ------------------------------------------------------------------
    if args.mindcube_json is not None:
        any_dataset = True
        out = (os.path.join(args.output_dir, "mindcube")
               if args.split_by_dataset else args.output_dir)
        writer = ParquetBufferedWriter(
            out, "mindcube", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=info_tracker,
        )
        csv_path = args.csv_output or os.path.join(args.output_dir, "mindcube_export.tsv")
        process_mindcube(
            args.mindcube_json,
            writer,
            mindcube_data_root=args.mindcube_data_root,
            csv_output_path=csv_path,
            hf_repo=args.hf_repo,
        )
        writer.finish()

    # ------------------------------------------------------------------
    #  OmniSpatial
    # ------------------------------------------------------------------
    if args.omnispatial_path is not None:
        any_dataset = True
        out = (os.path.join(args.output_dir, "omnispatial")
               if args.split_by_dataset else args.output_dir)
        writer = ParquetBufferedWriter(
            out, "omnispatial", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=info_tracker,
        )
        process_omnispatial(
            args.omnispatial_path,
            args.omnispatial_prompt_type,
            args.omnispatial_eval_type,
            writer,
        )
        writer.finish()

    # ------------------------------------------------------------------
    #  OST-Bench
    # ------------------------------------------------------------------
    if args.ost_file is not None:
        any_dataset = True
        if args.ost_image_root is None:
            print("[ERROR] --ost-image-root is required when "
                  "--ost-file is given")
            sys.exit(1)
        out = (os.path.join(args.output_dir, "ost")
               if args.split_by_dataset else args.output_dir)
        writer = ParquetBufferedWriter(
            out, "ost", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=info_tracker,
        )
        process_ost(args.ost_file, args.ost_image_root, writer)
        writer.finish()

    # ------------------------------------------------------------------
    #  SenseNova-SI  (produces two parquets: geo & und)
    # ------------------------------------------------------------------
    if args.sensenova_si_jsonl is not None:
        any_dataset = True
        if args.sensenova_si_image_root is None:
            print("[ERROR] --sensenova-si-image-root is required when "
                  "--sensenova-si-jsonl is given")
            sys.exit(1)

        # geo writer (with its own info tracker → separate parquet_info.json)
        geo_out = (os.path.join(args.output_dir, "sensenova_si_geo")
                   if args.split_by_dataset else args.output_dir)
        geo_info_tracker = ParquetInfoTracker(geo_out)
        geo_writer = ParquetBufferedWriter(
            geo_out, "sensenova_si_geo", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=geo_info_tracker,
        )
        # und writer (with its own info tracker → separate parquet_info.json)
        und_out = (os.path.join(args.output_dir, "sensenova_si_und")
                   if args.split_by_dataset else args.output_dir)
        und_info_tracker = ParquetInfoTracker(und_out)
        und_writer = ParquetBufferedWriter(
            und_out, "sensenova_si_und", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=und_info_tracker,
        )

        process_sensenova_si(
            args.sensenova_si_jsonl,
            args.sensenova_si_image_root,
            writer=geo_writer,
            min_images_per_sample=2,
            max_images_per_sample=16,
            task='geo'
        )
        geo_writer.finish()

        process_sensenova_si(
            args.sensenova_si_jsonl,
            args.sensenova_si_image_root,
            writer=geo_writer,
            min_images_per_sample=1,
            max_images_per_sample=16,
            task='und'
        )
        und_writer.finish()
        geo_info_tracker.summary()
        und_info_tracker.summary()

    # ------------------------------------------------------------------
    #  ARKitScenes
    # ------------------------------------------------------------------
    if args.arkitscenes_data_root is not None:
        any_dataset = True
        out = (os.path.join(args.output_dir, "arkitscenes")
               if args.split_by_dataset else args.output_dir)
        arkit_info_tracker = ParquetInfoTracker(out)
        writer = ParquetBufferedWriter(
            out, "arkitscenes", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=arkit_info_tracker,
        )
        process_arkitscenes(
            args.arkitscenes_data_root,
            writer,
            max_frames=args.arkitscenes_max_frames,
            frame_stride=args.arkitscenes_frame_stride,
            need_3d_annotation=args.need_3d_annotation,
        )
        writer.finish()
        arkit_info_tracker.summary()

    # ------------------------------------------------------------------
    #  RE10K
    # ------------------------------------------------------------------
    if args.re10k_data_root is not None:
        any_dataset = True
        out = (os.path.join(args.output_dir, "re10k")
               if args.split_by_dataset else args.output_dir)
        re10k_info_tracker = ParquetInfoTracker(out)
        writer = ParquetBufferedWriter(
            out, "re10k", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=re10k_info_tracker,
        )
        process_re10k(
            args.re10k_data_root,
            writer,
            min_frames=args.re10k_min_frames,
            max_frames=args.re10k_max_frames,
        )
        writer.finish()
        re10k_info_tracker.summary()
    
    # ------------------------------------------------------------------
    #  SPLD (SpatialLadder)
    # ------------------------------------------------------------------
    if args.spld_data_root is not None:
        any_dataset = True
        out = (os.path.join(args.output_dir, "spld")
               if args.split_by_dataset else args.output_dir)
        spld_info_tracker = ParquetInfoTracker(out)
        writer = ParquetBufferedWriter(
            out, "spld", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=spld_info_tracker,
        )
        process_spld(
            data_root=args.spld_data_root,
            writer=writer,
        )
        writer.finish()
        spld_info_tracker.summary()

    # ------------------------------------------------------------------
    #  VGLLM (VG-LLM-Data series)
    # ------------------------------------------------------------------
    if args.vgllm_json is not None:
        any_dataset = True
        # 解析 "json_path::data_dir" 格式
        vgllm_pairs: List[Tuple[str, str]] = []
        for entry in args.vgllm_json:
            if "::" in entry:
                parts = entry.split("::", 1)
                json_path = parts[0].strip()
                data_dir = parts[1].strip()
            else:
                print(f"[ERROR] --vgllm-json entry missing '::data_dir': {entry}")
                print("  Expected format: 'json_path::data_dir'")
                sys.exit(1)
            if not os.path.isfile(json_path):
                print(f"[WARN] JSON file not found: {json_path}")
            vgllm_pairs.append((json_path, data_dir))

        process_vgllm_series(
            json_data_dir_pairs=vgllm_pairs,
            output_dir=args.output_dir,
            split_by_dataset=args.split_by_dataset,
            rows_per_file=args.rows_per_file,
            rows_per_row_group=args.rows_per_row_group,
            num_workers=args.vgllm_num_workers,
            chunk_size=args.vgllm_chunk_size,
        )

    # ------------------------------------------------------------------
    #  ScanQA
    # ------------------------------------------------------------------
    if args.scanqa_json is not None:
        any_dataset = True
        if args.scanqa_embodiedscan_dir is None:
            print("[ERROR] --scanqa-embodiedscan-dir is required when "
                  "--scanqa-json is given")
            sys.exit(1)
        if args.scanqa_video_folder is None:
            print("[ERROR] --scanqa-video-folder is required when "
                  "--scanqa-json is given")
            sys.exit(1)

        out = (os.path.join(args.output_dir, "scanqa")
               if args.split_by_dataset else args.output_dir)
        scanqa_info_tracker = ParquetInfoTracker(out)
        writer = ParquetBufferedWriter(
            out, "scanqa", SCANQA_ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=scanqa_info_tracker,
        )
        process_scanqa(
            scanqa_json_files=args.scanqa_json,
            embodiedscan_dir=args.scanqa_embodiedscan_dir,
            video_folder=args.scanqa_video_folder,
            writer=writer,
            num_workers=args.scanqa_num_workers,
        )
        writer.finish()
        scanqa_info_tracker.summary()

    # ------------------------------------------------------------------
    #  Video3DLLM (VEGA-3D 五个数据集，task-grouped shuffle + 离线 3D 计算)
    # ------------------------------------------------------------------
    if args.video3dllm_json is not None:
        any_dataset = True
        if args.video3dllm_embodiedscan_dir is None:
            print("[ERROR] --video3dllm-embodiedscan-dir is required when "
                  "--video3dllm-json is given")
            sys.exit(1)
        if args.video3dllm_video_folder is None:
            print("[ERROR] --video3dllm-video-folder is required when "
                  "--video3dllm-json is given")
            sys.exit(1)

        # 解析 "json_path::data_root" 格式
        v3d_pairs: List[Tuple[str, str]] = []
        for entry in args.video3dllm_json:
            if "::" in entry:
                parts = entry.split("::", 1)
                json_path = parts[0].strip()
                data_root = parts[1].strip()
            else:
                json_path = entry.strip()
                data_root = ""
            if not os.path.isfile(json_path):
                print(f"[WARN] JSON file not found: {json_path}")
            v3d_pairs.append((json_path, data_root))

        scan2obj_files = args.video3dllm_scan2obj_files or []

        out = (os.path.join(args.output_dir, "video3dllm")
               if args.split_by_dataset else args.output_dir)

        process_video3dllm(
            json_data_dir_pairs=v3d_pairs,
            embodiedscan_dir=args.video3dllm_embodiedscan_dir,
            video_folder=args.video3dllm_video_folder,
            scan2obj_files=scan2obj_files,
            output_dir=out,
            rows_per_file=args.rows_per_file,
            rows_per_row_group=args.rows_per_row_group,
            shuffle_seed=args.video3dllm_shuffle_seed,
            num_workers=args.video3dllm_num_workers,
            chunk_size=args.video3dllm_chunk_size,
            max_samples=args.video3dllm_max_samples,
        )

    # ------------------------------------------------------------------
    #  Mixed MindCube + VGLLM
    # ------------------------------------------------------------------
    if args.mixed_mindcube_json is not None or args.mixed_vgllm_json is not None:
        any_dataset = True

        def _parse_mixed_entries(entries):
            """解析 'json_path::data_dir::repeat_time' 格式，返回 (json_path, data_dir, repeat_time) 列表。"""
            pairs = []
            if entries is None:
                return pairs
            for entry in entries:
                parts = entry.split("::")
                if len(parts) < 2:
                    print(f"[ERROR] Mixed entry missing '::data_dir': {entry}")
                    print("  Expected format: 'json_path::data_dir[::repeat_time]'")
                    sys.exit(1)
                json_path = parts[0].strip()
                data_dir = parts[1].strip()
                repeat_time = float(parts[2].strip()) if len(parts) >= 3 else 1.0
                if not os.path.isfile(json_path):
                    print(f"[WARN] JSON file not found: {json_path}")
                pairs.append((json_path, data_dir, repeat_time))
            return pairs

        mixed_mc_pairs = _parse_mixed_entries(args.mixed_mindcube_json)
        mixed_vgllm_pairs = _parse_mixed_entries(args.mixed_vgllm_json)

        mixed_out = (os.path.join(args.output_dir, "mixed_mc_vgllm")
                     if args.split_by_dataset else args.output_dir)

        process_mixed_mindcube_vgllm(
            mindcube_json_data_dir_pairs=mixed_mc_pairs,
            vgllm_json_data_dir_pairs=mixed_vgllm_pairs,
            output_dir=mixed_out,
            rows_per_file=args.rows_per_file,
            rows_per_row_group=args.rows_per_row_group,
            shuffle_seed=args.mixed_shuffle_seed,
            num_workers=args.mixed_num_workers,
            chunk_size=args.mixed_chunk_size,
            max_samples=args.mixed_max_samples,
        )

    # ------------------------------------------------------------------
    #  3DThinker-10K
    # ------------------------------------------------------------------
    if args.thinker3d_jsonl is not None:
        any_dataset = True
        if args.thinker3d_data_root is None:
            print("[ERROR] --thinker3d-data-root is required when "
                  "--thinker3d-jsonl is given")
            sys.exit(1)
        out = (os.path.join(args.output_dir, "3dthinker")
               if args.split_by_dataset else args.output_dir)
        thinker3d_info_tracker = ParquetInfoTracker(out)
        writer = ParquetBufferedWriter(
            out, "3dthinker", ARROW_SCHEMA,
            args.rows_per_file, args.rows_per_row_group,
            info_tracker=thinker3d_info_tracker,
        )
        process_3dthinker(
            args.thinker3d_jsonl,
            writer,
            data_root=args.thinker3d_data_root,
        )
        writer.finish()
        thinker3d_info_tracker.summary()

    # ------------------------------------------------------------------
    #  Ego3D-Bench → TSV
    # ------------------------------------------------------------------
    if args.ego3d_data_root is not None:
        any_dataset = True
        ego3d_tsv = args.ego3d_tsv_output or os.path.join(args.output_dir, "Ego3D-Bench.tsv")
        process_ego3d_to_tsv(
            data_root=args.ego3d_data_root,
            tsv_output_path=ego3d_tsv,
            hf_repo=args.ego3d_hf_repo,
        )

    # ------------------------------------------------------------------
    #  Finish
    # ------------------------------------------------------------------
    if not any_dataset:
        print("No datasets specified. Use --help for usage info.")
        sys.exit(1)

    info_tracker.summary()
    print("\n✅ All done.")


if __name__ == "__main__":
    main()