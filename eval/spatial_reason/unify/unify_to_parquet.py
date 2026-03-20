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
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    ("depth_list",      pa.list_(pa.string())),
    ("poses",           pa.list_(pa.list_(pa.float64()))),   # N × 16
    ("intrinsic",       pa.list_(pa.float64())),             # 16
    ("depth_intrinsic", pa.list_(pa.float64())),             # 16
    ("metadata",        pa.string()),
])

# 4×4 identity as 16-element list – used as placeholder for image-only datasets
_IDENTITY_4x4 = np.eye(4).flatten().tolist()


# ═════════════════════════════════════════════════════════════════════════
#  Per-dataset row converters
# ═════════════════════════════════════════════════════════════════════════

def _ensure_metadata_str(meta) -> str:
    """Normalise metadata to a JSON string."""
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
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
        "category":        "among",
        "dataset_name":    f"mindcube_{ds_name}",
        "image_path":      image_list,
        "depth_list":      [],
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
    if all_rows_for_csv:
        _save_csv_and_upload_hf(all_rows_for_csv, csv_output_path, hf_repo, dataset_label="mindcube")


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

    image_list = [os.path.join(scene_dir, f) for f in image_files]
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

    # ── Annotation Control ──
    p.add_argument("--need-3d-annotation", action="store_true", default=False,
        help="Whether to load 3D annotation data (poses, depth, intrinsics). "
            "If False, output directory will have 'no_3d_annotation' suffix."
    )

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
    #  Finish
    # ------------------------------------------------------------------
    if not any_dataset:
        print("No datasets specified. Use --help for usage info.")
        sys.exit(1)

    info_tracker.summary()
    print("\n✅ All done.")


if __name__ == "__main__":
    main()