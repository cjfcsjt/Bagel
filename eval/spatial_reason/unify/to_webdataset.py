#!/usr/bin/env python3
"""
Convert ARKitScenes Training data into **WebDataset** shards and
(optionally) upload them to a HuggingFace Hub dataset repository.

Features
--------
* Handles both **extracted** scenes (``lowres_wide/`` directory exists) and
  **zipped-only** scenes (``lowres_wide.zip`` etc.) — zips are read in-memory
  without permanently extracting to disk.
* Each WebDataset sample corresponds to **one scene** and contains:
    - ``vga_wide/{fname}``                  : VGA-wide RGB frames (original filenames)
    - ``lowres_depth/{fname}``               : depth maps (if available)
    - ``vga_wide_intrinsics/{fname}``       : raw .pincam intrinsic files
    - ``lowres_wide.traj``                  : raw trajectory file (poses)
    - ``annotation.json``                   : 3D object detection annotations (if available)
    - ``mesh.ply``                          : 3D mesh (if available)
    - ``meta.json``                         : scene id, number of frames, etc.
* All data is preserved as-is from the original dataset — no parsing or transformation.
* Shards are written as ``.tar`` files with configurable max samples per shard.

Usage
-----
python to_webdataset.py \\
    --data-root /path/to/ARKitScenes/ar_raw_all/raw \\
    --output-dir /path/to/wds_output \\
    --max-samples-per-shard 50 \\
    --hf-repo your-username/arkitscenes-wds \\
    --num-workers 8

The script automatically discovers Training/ and Validation/ subdirectories
under ``--data-root`` and writes shards to ``{output-dir}/train/`` and
``{output-dir}/val/`` respectively.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import itertools
import json
import os
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  Scene data loader — handles both extracted dirs and zip archives
# ═══════════════════════════════════════════════════════════════════════

def _load_files_from_dir(directory: str, ext: str = ".png") -> Dict[str, bytes]:
    """Load all files with given extension from an extracted directory."""
    result = {}
    for fname in os.listdir(directory):
        if fname.lower().endswith(ext):
            fpath = os.path.join(directory, fname)
            with open(fpath, "rb") as f:
                result[fname] = f.read()
    return result


def _load_files_from_zip(zip_path: str, ext: str = ".png") -> Dict[str, bytes]:
    """Load all files with given extension from a zip archive (in-memory)."""
    result = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            basename = os.path.basename(name)
            if basename and basename.lower().endswith(ext):
                result[basename] = zf.read(name)
    return result


def _load_data_source(
    scene_dir: str, sub_name: str, ext: str = ".png"
) -> Dict[str, bytes]:
    """
    Load files from either an extracted sub-directory or a zip archive.
    e.g. sub_name="lowres_wide" looks for ``scene_dir/lowres_wide/`` first,
    then falls back to ``scene_dir/lowres_wide.zip``.
    """
    dir_path = os.path.join(scene_dir, sub_name)
    zip_path = os.path.join(scene_dir, f"{sub_name}.zip")

    if os.path.isdir(dir_path):
        return _load_files_from_dir(dir_path, ext)
    elif os.path.isfile(zip_path):
        return _load_files_from_zip(zip_path, ext)
    else:
        return {}


def process_single_scene(
    scene_dir: str,
    scene_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Process a single ARKitScenes scene and return a dict ready to be
    written into a WebDataset tar shard.

    All frames are kept as-is (no sorting, striding, or sub-sampling).

    Returns None on failure (no images found, etc.).

    The returned dict has keys:
        __key__                              : str (scene_id)
        meta.json                            : bytes (JSON metadata)
        vga_wide/{fname}                     : bytes (VGA-wide RGB frames, original filename)
        lowres_depth/{fname}                 : bytes (depth frames, if available)
        vga_wide_intrinsics/{fname}          : bytes (raw .pincam intrinsic files)
        lowres_wide.traj                     : bytes (raw trajectory file)
        annotation.json                      : bytes (3D OD annotation, if exists)
        mesh.ply                             : bytes (3D mesh, if exists)
    """
    # ── Load RGB images (from vga_wide) ──
    rgb_files = _load_data_source(scene_dir, "vga_wide", ".png")
    if not rgb_files:
        return None

    # Keep original filenames, just sort for deterministic order
    rgb_fnames = sorted(rgb_files.keys())
    n_frames = len(rgb_fnames)

    # ── Load depth images ──
    depth_files = _load_data_source(scene_dir, "lowres_depth", ".png")

    # ── Load intrinsics (.pincam) from vga_wide_intrinsics — raw bytes ──
    intrinsics_files = _load_data_source(
        scene_dir, "vga_wide_intrinsics", ".pincam"
    )

    # ── Build sample dict ──
    sample: Dict[str, Any] = {"__key__": scene_id}

    # VGA-wide RGB frames — preserve original filenames
    for fname in rgb_fnames:
        sample[f"vga_wide/{fname}"] = rgb_files[fname]

    # Depth frames — preserve original filenames
    for fname in rgb_fnames:
        if fname in depth_files:
            sample[f"lowres_depth/{fname}"] = depth_files[fname]

    # VGA-wide intrinsics (.pincam) — raw files preserved
    for pincam_name, pincam_bytes in intrinsics_files.items():
        sample[f"vga_wide_intrinsics/{pincam_name}"] = pincam_bytes

    # 3D object detection annotation (if available)
    anno_path = os.path.join(scene_dir, f"{scene_id}_3dod_annotation.json")
    if os.path.isfile(anno_path):
        with open(anno_path, "rb") as f:
            sample["annotation.json"] = f.read()

    # 3D mesh (if available)
    mesh_path = os.path.join(scene_dir, f"{scene_id}_3dod_mesh.ply")
    if os.path.isfile(mesh_path):
        with open(mesh_path, "rb") as f:
            sample["mesh.ply"] = f.read()

    # Trajectory file (raw, if available)
    traj_path = os.path.join(scene_dir, "lowres_wide.traj")
    if os.path.isfile(traj_path):
        with open(traj_path, "rb") as f:
            sample["lowres_wide.traj"] = f.read()

    # Metadata
    meta = {
        "scene_id": scene_id,
        "n_frames": n_frames,
        "rgb_filenames": rgb_fnames,
        "has_depth": bool(depth_files),
        "has_poses": os.path.isfile(traj_path),
        "has_intrinsics": bool(intrinsics_files),
        "has_annotation": os.path.isfile(anno_path),
        "has_mesh": os.path.isfile(mesh_path),
    }
    sample["meta.json"] = json.dumps(meta).encode("utf-8")

    return sample


# ═══════════════════════════════════════════════════════════════════════
#  WebDataset shard writer
# ═══════════════════════════════════════════════════════════════════════

class ShardWriter:
    """
    Write samples to WebDataset-style tar shards.

    Each shard is named ``{prefix}-{shard_idx:06d}.tar`` and contains
    at most ``max_samples`` samples.
    """

    def __init__(self, output_dir: str, prefix: str = "arkitscenes",
                 max_samples: int = 50):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_samples = max_samples

        os.makedirs(output_dir, exist_ok=True)
        self.shard_idx = 0
        self.sample_count = 0
        self.total_samples = 0
        self.tar: Optional[tarfile.TarFile] = None
        self.shard_paths: List[str] = []
        # Track which scenes are in each shard
        self._current_shard_scenes: List[str] = []
        self._shard_scene_map: Dict[str, List[str]] = {}  # shard_name -> [scene_ids]
        self._scene_shard_map: Dict[str, str] = {}  # scene_id -> shard_name
        self._open_new_shard()

    def _open_new_shard(self):
        """Open a new tar shard file."""
        if self.tar is not None:
            self.tar.close()
            # Save the scene list for the shard we just closed
            prev_shard = os.path.basename(self.shard_paths[-1])
            self._shard_scene_map[prev_shard] = list(self._current_shard_scenes)
            self._current_shard_scenes = []
        shard_name = f"{self.prefix}-{self.shard_idx:06d}.tar"
        shard_path = os.path.join(self.output_dir, shard_name)
        self.tar = tarfile.open(shard_path, "w")
        self.shard_paths.append(shard_path)
        self.sample_count = 0

    def add_sample(self, sample: Dict[str, Any]):
        """
        Add one sample (scene) to the current shard.
        Automatically rolls over to a new shard if limit reached.
        """
        if self.sample_count >= self.max_samples:
            self.shard_idx += 1
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
        """Close the current shard and write shards.txt + metadata.json."""
        if self.tar is not None:
            self.tar.close()
            self.tar = None
            # Save the scene list for the last shard
            if self.shard_paths:
                last_shard = os.path.basename(self.shard_paths[-1])
                self._shard_scene_map[last_shard] = list(self._current_shard_scenes)
                self._current_shard_scenes = []

        if self.total_samples == 0:
            return

        # ── Write shards.txt ──
        # Format: one line per shard, "shard_name\tscene1,scene2,..."
        shards_txt_path = os.path.join(self.output_dir, "shards.txt")
        with open(shards_txt_path, "w") as f:
            for shard_path in self.shard_paths:
                shard_name = os.path.basename(shard_path)
                scenes = self._shard_scene_map.get(shard_name, [])
                f.write(f"{shard_name}\t{','.join(scenes)}\n")

        # ── Compute per-shard file sizes and checksums ──
        shard_info_list = []
        for shard_path in self.shard_paths:
            shard_name = os.path.basename(shard_path)
            file_size = os.path.getsize(shard_path) if os.path.isfile(shard_path) else 0
            # Compute MD5 for integrity checking
            md5 = hashlib.md5()
            if os.path.isfile(shard_path):
                with open(shard_path, "rb") as sf:
                    for chunk in iter(lambda: sf.read(8192), b""):
                        md5.update(chunk)
            shard_info_list.append({
                "shard_name": shard_name,
                "num_scenes": len(self._shard_scene_map.get(shard_name, [])),
                "scenes": self._shard_scene_map.get(shard_name, []),
                "file_size_bytes": file_size,
                "md5": md5.hexdigest(),
            })

        # ── Write metadata.json ──
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_scenes": self.total_samples,
            "total_shards": len(self.shard_paths),
            "max_samples_per_shard": self.max_samples,
            "shards": shard_info_list,
            "scene_to_shard": self._scene_shard_map,
        }
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Wrote {shards_txt_path}")
        print(f"  Wrote {metadata_path}")

    def summary(self) -> str:
        return (
            f"Wrote {self.total_samples} samples across "
            f"{len(self.shard_paths)} shard(s) to {self.output_dir}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  HuggingFace upload
# ═══════════════════════════════════════════════════════════════════════

def upload_to_hf(output_dir: str, repo_id: str, private: bool = False):
    """Upload all shard .tar files (for all splits) to a HuggingFace Hub dataset repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"[HF] Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"[HF] Warning creating repo: {e}")

    # Discover split subdirectories (train/, val/) and upload each
    split_dirs = sorted(
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    )
    if not split_dirs:
        # Fallback: output_dir itself contains tar files (single split)
        split_dirs = ["."]

    # Also upload shards.txt and metadata.json per split

    total_uploaded = 0
    for split_name in split_dirs:
        split_path = os.path.join(output_dir, split_name)
        tar_files = sorted(
            f for f in os.listdir(split_path) if f.endswith(".tar")
        )
        if not tar_files:
            continue

        remote_prefix = split_name if split_name != "." else "data"
        print(f"[HF] Uploading {len(tar_files)} shard(s) for split '{split_name}' to {repo_id} ...")

        for fname in tqdm(tar_files, desc=f"Uploading {split_name}"):
            fpath = os.path.join(split_path, fname)
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"{remote_prefix}/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            total_uploaded += 1

        # Upload shards.txt and metadata.json for this split
        for extra_file in ["shards.txt", "metadata.json"]:
            extra_path = os.path.join(split_path, extra_file)
            if os.path.isfile(extra_path):
                api.upload_file(
                    path_or_fileobj=extra_path,
                    path_in_repo=f"{remote_prefix}/{extra_file}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                total_uploaded += 1

    # Upload a README
    readme_content = f"""---
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/*.tar
      - split: validation
        path: val/*.tar
---

# ARKitScenes WebDataset

This dataset contains **ARKitScenes** data (Training + Validation) converted to
[WebDataset](https://github.com/webdataset/webdataset) format.

Each sample is one indoor scene with:
- ``vga_wide/`` — VGA-wide RGB frames
- ``lowres_depth/`` — depth maps
- ``vga_wide_intrinsics/`` — raw .pincam intrinsic files
- ``lowres_wide.traj`` — raw trajectory file (poses)
- ``annotation.json`` — 3D object detection annotations
- ``mesh.ply`` — 3D mesh
- ``meta.json`` — scene metadata
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] Upload complete ({total_uploaded} files): https://huggingface.co/datasets/{repo_id}")


# ═══════════════════════════════════════════════════════════════════════
#  Worker function for parallel processing
# ═══════════════════════════════════════════════════════════════════════

def _worker_process_scene(args_tuple):
    """Worker function for ProcessPoolExecutor."""
    scene_dir, scene_id = args_tuple
    try:
        return process_single_scene(scene_dir, scene_id)
    except Exception as e:
        print(f"[WARN] Failed to process scene {scene_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert ARKitScenes to WebDataset shards and "
                    "optionally upload to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to ARKitScenes raw directory containing Training/ "
             "and/or Validation/ (e.g. .../ar_raw_all/raw)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write WebDataset .tar shards",
    )
    parser.add_argument(
        "--max-samples-per-shard", type=int, default=50,
        help="Maximum number of scenes per .tar shard",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo ID to upload to "
             "(e.g. 'username/arkitscenes-wds'). "
             "If not specified, only local shards are written.",
    )
    parser.add_argument(
        "--hf-private", action="store_true", default=False,
        help="Make the HuggingFace repo private",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of parallel workers for scene processing "
             "(0 = sequential, no multiprocessing)",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)

    # ── Discover splits (Training / Validation) ──
    split_map = {
        "Training": "train",
        "Validation": "val",
    }
    splits_to_process = []
    for orig_name, short_name in split_map.items():
        split_dir = os.path.join(data_root, orig_name)
        if os.path.isdir(split_dir):
            splits_to_process.append((orig_name, short_name, split_dir))

    if not splits_to_process:
        # Fallback: treat data_root itself as a single split
        print(f"[INFO] No Training/Validation subdirs found; "
              f"treating data-root as a single split.")
        splits_to_process = [("data", "data", data_root)]

    print(f"[ARKitScenes → WebDataset]")
    print(f"  Data root    : {data_root}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Splits found : {[s[0] for s in splits_to_process]}")
    print(f"  Samples/shard: {args.max_samples_per_shard}")
    print(f"  Workers      : {args.num_workers or 'sequential'}")
    print()

    grand_processed = 0
    grand_skipped = 0

    for orig_name, short_name, split_dir in splits_to_process:
        # ── Discover scenes for this split ──
        scene_ids = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        print(f"\n{'='*60}")
        print(f"  Split: {orig_name} → {short_name}/  ({len(scene_ids)} scenes)")
        print(f"{'='*60}")

        # Output to {output_dir}/{short_name}/
        split_output_dir = os.path.join(args.output_dir, short_name)

        # ── Create shard writer ──
        writer = ShardWriter(
            split_output_dir,
            prefix="arkitscenes",
            max_samples=args.max_samples_per_shard,
        )

        processed = 0
        skipped = 0

        # if args.num_workers > 0:
        #     # ── Parallel processing with bounded in-flight tasks ──
        #     # Limit concurrent futures to avoid memory buildup from
        #     # completed-but-unconsumed results.
        #     tasks = [
        #         (os.path.join(split_dir, sid), sid)
        #         for sid in scene_ids
        #     ]
        #     max_inflight = args.num_workers * 2
        #     task_iter = iter(tasks)
        #     total_tasks = len(tasks)

        #     with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        #         # Seed the initial batch of futures
        #         futures = {}
        #         for t in itertools.islice(task_iter, max_inflight):
        #             fut = executor.submit(_worker_process_scene, t)
        #             futures[fut] = t[1]

        #         pbar = tqdm(total=total_tasks, desc=f"Processing {orig_name}")
        #         while futures:
        #             done, _ = concurrent.futures.wait(
        #                 futures,
        #                 return_when=concurrent.futures.FIRST_COMPLETED,
        #             )
        #             for future in done:
        #                 scene_id = futures.pop(future)
        #                 try:
        #                     sample = future.result()
        #                 except Exception as e:
        #                     print(f"[WARN] Scene {scene_id} raised: {e}")
        #                     sample = None

        #                 if sample is None:
        #                     skipped += 1
        #                 else:
        #                     writer.add_sample(sample)
        #                     del sample  # Explicitly free large scene data
        #                     processed += 1

        #                 pbar.update(1)
        #                 pbar.set_postfix(ok=processed, skip=skipped)

        #                 # Replenish: submit a new task for each completed one
        #                 next_task = next(task_iter, None)
        #                 if next_task is not None:
        #                     fut = executor.submit(_worker_process_scene, next_task)
        #                     futures[fut] = next_task[1]
        #         pbar.close()
        # else:
        #     # ── Sequential processing ──
        #     for scene_id in tqdm(scene_ids, desc=f"Processing {orig_name}"):
        #         scene_dir = os.path.join(split_dir, scene_id)
        #         sample = process_single_scene(scene_dir, scene_id)
        #         if sample is None:
        #             skipped += 1
        #             continue
        #         writer.add_sample(sample)
        #         del sample  # Explicitly free large scene data
        #         processed += 1

        writer.close()

        print(f"\n[{orig_name}] {writer.summary()}")
        print(f"  Processed: {processed}, Skipped: {skipped}")

        grand_processed += processed
        grand_skipped += skipped

    print(f"\n{'='*60}")
    print(f"[All Done] Total processed: {grand_processed}, skipped: {grand_skipped}")
    print(f"{'='*60}")

    # ── Upload to HuggingFace ──
    if args.hf_repo:
        upload_to_hf(args.output_dir, args.hf_repo, private=args.hf_private)


if __name__ == "__main__":
    main()
