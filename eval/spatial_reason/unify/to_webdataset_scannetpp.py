#!/usr/bin/env python3
"""
Convert ScanNet++ data into **WebDataset** shards and
(optionally) upload them to a HuggingFace Hub dataset repository.

Features
--------
* Reads the official ScanNet++ directory layout at
  ``{data-root}/data/{scene_id}/{dslr,iphone,scans,panocam}/``.
* Processes **all** scenes in ``data/`` at once (no train/val splitting).
* Split files from ``{data-root}/splits/`` are copied into the output
  directory so they are uploaded together to HuggingFace.
* Each WebDataset sample corresponds to **one scene** and contains the
  **entire** directory tree of that scene, preserving all subdirectories
  and files as-is.  Only ``.mkv`` / ``.bin`` files are skipped (too large).
  Mesh ``.ply`` files are controlled by ``--include-mesh`` / ``--no-mesh``,
  and the ``iphone/`` subtree by ``--include-iphone``.
* Shards are written as ``.tar`` files with configurable max samples per shard.

Usage
-----
python to_webdataset_scannetpp.py \\
    --data-root /dfs/dataset/gui_dataset/scannetpp/scannetpp \\
    --output-dir /path/to/wds_output \\
    --max-samples-per-shard 10 \\
    --num-workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  Scene data loader
# ═══════════════════════════════════════════════════════════════════════

def _load_dir_recursive(directory: str) -> Dict[str, bytes]:
    """
    Recursively load ALL files under *directory*.
    Returns a dict mapping relative paths (from *directory*) to bytes.
    """
    result: Dict[str, bytes] = {}
    if not os.path.isdir(directory):
        return result
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, directory)
            with open(fpath, "rb") as f:
                result[rel] = f.read()
    return result


# Files to skip (too large for WebDataset shards)
# - .mkv: iPhone video files (~300MB each)
# - .bin: iPhone depth data (~500MB each)
# These CAN be read/restored perfectly, but are skipped by default to
# keep shard sizes and memory usage manageable.
# Set to empty set() to include everything.
_SKIP_EXTENSIONS = {} # {".mkv", ".bin"}


def process_single_scene(
    scene_dir: str,
    scene_id: str,
    include_mesh: bool = True,
    include_iphone: bool = False,
    max_dslr_images: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Process a single ScanNet++ scene by reading the **entire** directory
    tree and returning a dict ready to be written into a WebDataset tar
    shard.

    The scene directory is walked recursively.  Every file is included
    with its original relative path preserved, except:
      - ``.mkv`` / ``.bin`` files (too large, skipped by default)
      - 3D mesh ``.ply`` files are only included when *include_mesh* is True
      - ``iphone/`` subtree is only included when *include_iphone* is True

    When *max_dslr_images* > 0 the DSLR image directories
    (``resized_undistorted_images`` / ``resized_images``) are subsampled.

    Returns None when the scene has no DSLR images at all.
    Returns (None, reason_string) tuple when skipped, or (sample, None) on success.
    """
    if not os.path.isdir(scene_dir):
        return None, f"scene directory does not exist: {scene_dir}"

    sample: Dict[str, Any] = {"__key__": scene_id}
    n_frames = 0
    dslr_filenames: List[str] = []

    for root, dirs, files in os.walk(scene_dir):
        # relative path from scene_dir  e.g. "dslr/resized_undistorted_images"
        rel_root = os.path.relpath(root, scene_dir)
        if rel_root == ".":
            rel_root = ""

        # ── Optionally skip iPhone subtree ──
        if not include_iphone and rel_root.startswith("iphone"):
            dirs.clear()  # don't descend further
            continue

        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            # Skip large binary blobs
            if ext in _SKIP_EXTENSIONS:
                continue

            # Skip mesh .ply files if not requested
            if not include_mesh and ext == ".ply":
                continue

            rel_path = os.path.join(rel_root, fname) if rel_root else fname

            with open(fpath, "rb") as f:
                sample[rel_path] = f.read()

    # ── Figure out DSLR image count (for metadata & optional subsampling) ──
    dslr_img_dir = "dslr/resized_undistorted_images"
    dslr_keys = sorted(
        k for k in sample
        if k.startswith(dslr_img_dir + "/")
    )
    if not dslr_keys:
        # Fallback: try resized_images
        dslr_img_dir = "dslr/resized_images"
        dslr_keys = sorted(
            k for k in sample
            if k.startswith(dslr_img_dir + "/")
        )

    if not dslr_keys:
        # No DSLR images at all → skip this scene
        return None, f"no DSLR images found in dslr/resized_undistorted_images/ or dslr/resized_images/"

    # ── Subsample DSLR images if requested ──
    if max_dslr_images > 0 and len(dslr_keys) > max_dslr_images:
        step = len(dslr_keys) / max_dslr_images
        keep_keys = {dslr_keys[int(i * step)] for i in range(max_dslr_images)}
        # Also subsample corresponding masks / anon_masks by matching basenames
        keep_basenames = {
            os.path.splitext(os.path.basename(k))[0] for k in keep_keys
        }
        # Directories whose images should be subsampled together
        _img_dirs = (
            "dslr/resized_undistorted_images/",
            "dslr/resized_images/",
            "dslr/resized_undistorted_masks/",
            "dslr/resized_anon_masks/",
        )
        keys_to_remove = []
        for k in list(sample.keys()):
            if any(k.startswith(d) for d in _img_dirs):
                basename_no_ext = os.path.splitext(os.path.basename(k))[0]
                if basename_no_ext not in keep_basenames:
                    keys_to_remove.append(k)
        for k in keys_to_remove:
            del sample[k]
        dslr_keys = sorted(
            k for k in sample
            if k.startswith(dslr_img_dir + "/")
        )

    dslr_filenames = [os.path.basename(k) for k in dslr_keys]
    n_frames = len(dslr_filenames)

    # ── Collect sub-directory names present ──
    top_dirs = {k.split("/")[0] for k in sample if "/" in k and k != "__key__"}

    # ── Write metadata ──
    meta = {
        "scene_id": scene_id,
        "n_dslr_frames": n_frames,
        "dslr_filenames": dslr_filenames,
        "subdirs": sorted(top_dirs),
    }
    sample["meta.json"] = json.dumps(meta).encode("utf-8")

    return sample, None


# ═══════════════════════════════════════════════════════════════════════
#  WebDataset shard writer
# ═══════════════════════════════════════════════════════════════════════

class ShardWriter:
    """
    Write samples to WebDataset-style tar shards.

    Each shard is named ``{prefix}-{shard_idx:06d}.tar`` and contains
    at most ``max_samples`` samples.

    Supports **incremental mode**: if a ``metadata.json`` already exists
    in *output_dir*, the writer will pick up from where it left off —
    preserving existing shards and only writing new ones.
    """

    def __init__(self, output_dir: str, prefix: str = "scannetpp",
                 max_samples: int = 10):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_samples = max_samples

        os.makedirs(output_dir, exist_ok=True)

        # ── Try to resume from existing metadata ──
        self._existing_shard_paths: List[str] = []
        self._existing_shard_scene_map: Dict[str, List[str]] = {}
        self._existing_scene_shard_map: Dict[str, str] = {}
        self._existing_total_samples: int = 0

        metadata_path = os.path.join(output_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                prev_meta = json.load(f)
            # Restore previous state
            for shard_info in prev_meta.get("shards", []):
                shard_name = shard_info["shard_name"]
                shard_path = os.path.join(output_dir, shard_name)
                if os.path.isfile(shard_path):
                    self._existing_shard_paths.append(shard_path)
                    self._existing_shard_scene_map[shard_name] = shard_info.get("scenes", [])
            self._existing_scene_shard_map = dict(prev_meta.get("scene_to_shard", {}))
            self._existing_total_samples = prev_meta.get("total_scenes", 0)
            # Next shard index = max existing + 1
            next_idx = len(self._existing_shard_paths)
            print(f"[Incremental] Found existing metadata with "
                  f"{self._existing_total_samples} scenes in "
                  f"{len(self._existing_shard_paths)} shard(s). "
                  f"New shards will start at index {next_idx}.")
        else:
            next_idx = 0

        self.shard_idx = next_idx
        self.sample_count = 0
        self.total_samples = 0  # counts only NEW samples in this run
        self.tar: Optional[tarfile.TarFile] = None
        self.shard_paths: List[str] = []  # only NEW shard paths
        self._current_shard_scenes: List[str] = []
        self._shard_scene_map: Dict[str, List[str]] = {}
        self._scene_shard_map: Dict[str, str] = {}
        self._new_shard_opened = False  # defer opening until first sample

    def get_existing_scene_ids(self) -> set:
        """Return the set of scene IDs already present in existing shards."""
        return set(self._existing_scene_shard_map.keys())

    def _open_new_shard(self):
        """Open a new tar shard file."""
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
        """
        Add one sample (scene) to the current shard.
        Automatically rolls over to a new shard if limit reached.
        Lazily opens the first shard on the first call.
        """
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
        """Close the current shard and write shards.txt + metadata.json,
        merging with existing shard information for incremental runs."""
        if self.tar is not None:
            self.tar.close()
            self.tar = None
            if self.shard_paths:
                last_shard = os.path.basename(self.shard_paths[-1])
                self._shard_scene_map[last_shard] = list(self._current_shard_scenes)
                self._current_shard_scenes = []

        # ── Merge existing and new shard information ──
        all_shard_paths = self._existing_shard_paths + self.shard_paths
        all_shard_scene_map = dict(self._existing_shard_scene_map)
        all_shard_scene_map.update(self._shard_scene_map)
        all_scene_shard_map = dict(self._existing_scene_shard_map)
        all_scene_shard_map.update(self._scene_shard_map)
        all_total = self._existing_total_samples + self.total_samples

        if all_total == 0:
            return

        # ── Write shards.txt ──
        shards_txt_path = os.path.join(self.output_dir, "shards.txt")
        with open(shards_txt_path, "w") as f:
            for shard_path in all_shard_paths:
                shard_name = os.path.basename(shard_path)
                scenes = all_shard_scene_map.get(shard_name, [])
                f.write(f"{shard_name}\t{','.join(scenes)}\n")

        # ── Compute per-shard file sizes ──
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

        # ── Write metadata.json ──
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

        print(f"  Wrote {shards_txt_path}")
        print(f"  Wrote {metadata_path}")

    def summary(self) -> str:
        all_total = self._existing_total_samples + self.total_samples
        all_shards = len(self._existing_shard_paths) + len(self.shard_paths)
        parts = [
            f"Total: {all_total} scenes across {all_shards} shard(s) "
            f"in {self.output_dir}"
        ]
        if self._existing_total_samples > 0:
            parts.append(
                f"  (existing: {self._existing_total_samples} scenes in "
                f"{len(self._existing_shard_paths)} shard(s), "
                f"new: {self.total_samples} scenes in "
                f"{len(self.shard_paths)} shard(s))"
            )
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  HuggingFace upload
# ═══════════════════════════════════════════════════════════════════════

def upload_to_hf(output_dir: str, repo_id: str, private: bool = False):
    """Upload all files in output_dir to a HuggingFace Hub dataset repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"[HF] Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"[HF] Warning creating repo: {e}")

    # Collect all files to upload
    files_to_upload = []
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            files_to_upload.append((fpath, fname))

    if not files_to_upload:
        print("[HF] No files found to upload.")
        return

    print(f"[HF] Uploading {len(files_to_upload)} file(s) ...")

    for fpath, fname in tqdm(files_to_upload, desc="Uploading"):
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
        path: "*.tar"
---

# ScanNet++ WebDataset

This dataset contains **ScanNet++** data converted to
[WebDataset](https://github.com/webdataset/webdataset) format.

All scenes are stored in flat `.tar` shards. Split files
(`nvs_sem_train.txt`, `nvs_sem_val.txt`, etc.) are included
alongside the shards for downstream use.

Each sample is one indoor scene with:
- ``dslr/resized_undistorted_images/`` — undistorted DSLR RGB images
- ``dslr/resized_undistorted_masks/``  — undistortion masks
- ``dslr/nerfstudio/``                 — nerfstudio-format camera data
- ``dslr/colmap/``                     — COLMAP camera/image data
- ``dslr/train_test_lists.json``       — per-scene train/test splits
- ``scans/mesh_aligned_0.05.ply``      — 3D mesh
- ``scans/mesh_aligned_0.05_semantic.ply`` — semantic mesh
- ``scans/segments_anno.json``         — semantic annotations
- ``iphone/``                          — iPhone pose/intrinsic data (optional)
- ``meta.json``                        — scene metadata
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] Upload complete ({len(files_to_upload)} files): "
          f"https://huggingface.co/datasets/{repo_id}")


# ═══════════════════════════════════════════════════════════════════════
#  Worker for multiprocessing (must be at module level to be picklable)
# ═══════════════════════════════════════════════════════════════════════

def _worker(args_tuple):
    sd, sid, im, ii, mdi = args_tuple
    try:
        return process_single_scene(sd, sid, im, ii, mdi)
    except Exception as e:
        print(f"[WARN] Failed to process scene {sid}: {e}")
        return None, f"exception: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert ScanNet++ to WebDataset shards and "
                    "optionally upload to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str,
        default="/dfs/dataset/gui_dataset/scannetpp/scannetpp",
        help="Path to the ScanNet++ root directory "
             "(containing data/, splits/, metadata/)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write WebDataset .tar shards",
    )
    parser.add_argument(
        "--max-samples-per-shard", type=int, default=10,
        help="Maximum number of scenes per .tar shard "
             "(ScanNet++ scenes are large, keep this small)",
    )
    parser.add_argument(
        "--max-dslr-images", type=int, default=0,
        help="Max DSLR images per scene (0 = all). "
             "Useful to reduce shard size.",
    )
    parser.add_argument(
        "--include-mesh", action="store_true", default=True,
        help="Include 3D mesh .ply files (can be ~80MB+ per scene)",
    )
    parser.add_argument(
        "--no-mesh", action="store_true", default=False,
        help="Exclude 3D mesh .ply files to reduce shard size",
    )
    parser.add_argument(
        "--include-iphone", action="store_true", default=False,
        help="Include iPhone pose/intrinsic data "
             "(video .mkv is never included due to size)",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo ID to upload to",
    )
    parser.add_argument(
        "--hf-private", action="store_true", default=False,
        help="Make the HuggingFace repo private",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of parallel workers (0 = sequential). "
             "Note: parallel processing loads entire scenes into memory.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)

    include_mesh = args.include_mesh and not args.no_mesh

    data_dir = os.path.join(data_root, "data")
    splits_dir = os.path.join(data_root, "splits")

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Copy split files to output directory ──
    if os.path.isdir(splits_dir):
        split_files = [f for f in os.listdir(splits_dir) if f.endswith(".txt")]
        for sf in split_files:
            src = os.path.join(splits_dir, sf)
            dst = os.path.join(args.output_dir, sf)
            shutil.copy2(src, dst)
            print(f"  Copied split file: {sf}")
    else:
        print(f"[WARN] Splits directory not found: {splits_dir}")

    # ── Discover all scenes ──
    all_scene_ids = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not all_scene_ids:
        print(f"[ERROR] No scene directories found in {data_dir}")
        sys.exit(1)

    print(f"[ScanNet++ → WebDataset]")
    print(f"  Data root      : {data_root}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Total scenes   : {len(all_scene_ids)}")
    print(f"  Samples/shard  : {args.max_samples_per_shard}")
    print(f"  Include mesh   : {include_mesh}")
    print(f"  Include iPhone : {args.include_iphone}")
    print(f"  Max DSLR imgs  : {args.max_dslr_images or 'all'}")
    print(f"  Workers        : {args.num_workers or 'sequential'}")
    print()

    writer = ShardWriter(
        args.output_dir,
        prefix="scannetpp",
        max_samples=args.max_samples_per_shard,
    )

    # ── Incremental: filter out already-processed scenes ──
    existing_scenes = writer.get_existing_scene_ids()
    if existing_scenes:
        new_scene_ids = [s for s in all_scene_ids if s not in existing_scenes]
        print(f"[Incremental] {len(existing_scenes)} scenes already in shards, "
              f"{len(new_scene_ids)} new scenes to process "
              f"(out of {len(all_scene_ids)} total).")
    else:
        new_scene_ids = all_scene_ids

    processed = 0
    skipped = 0
    already_done = len(all_scene_ids) - len(new_scene_ids)

    # if not new_scene_ids:
    #     print("[Incremental] Nothing new to process.")
    # elif args.num_workers > 0:
    #     from concurrent.futures import ProcessPoolExecutor, as_completed
    #     import itertools
    #     import concurrent.futures

    #     tasks = [
    #         (os.path.join(data_dir, sid), sid, include_mesh,
    #          args.include_iphone, args.max_dslr_images)
    #         for sid in new_scene_ids
    #     ]
    #     max_inflight = args.num_workers * 2
    #     task_iter = iter(tasks)

    #     with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    #         futures = {}
    #         for t in itertools.islice(task_iter, max_inflight):
    #             fut = executor.submit(_worker, t)
    #             futures[fut] = t[1]  # scene_id

    #         pbar = tqdm(total=len(tasks), desc="Processing scenes")
    #         while futures:
    #             done, _ = concurrent.futures.wait(
    #                 futures,
    #                 return_when=concurrent.futures.FIRST_COMPLETED,
    #             )
    #             for future in done:
    #                 scene_id = futures.pop(future)
    #                 try:
    #                     result = future.result()
    #                 except Exception as e:
    #                     print(f"[WARN] Scene {scene_id} raised: {e}")
    #                     result = (None, f"exception: {e}")

    #                 sample, reason = result
    #                 if sample is None:
    #                     skipped += 1
    #                     print(f"[SKIP] {scene_id}: {reason}")
    #                 else:
    #                     writer.add_sample(sample)
    #                     del sample
    #                     processed += 1

    #                 pbar.update(1)
    #                 pbar.set_postfix(ok=processed, skip=skipped)

    #                 next_task = next(task_iter, None)
    #                 if next_task is not None:
    #                     fut = executor.submit(_worker, next_task)
    #                     futures[fut] = next_task[1]
    #         pbar.close()
    # else:
    #     # ── Sequential processing ──
    #     for scene_id in tqdm(new_scene_ids, desc="Processing scenes"):
    #         scene_dir = os.path.join(data_dir, scene_id)
    #         sample, reason = process_single_scene(
    #             scene_dir, scene_id,
    #             include_mesh=include_mesh,
    #             include_iphone=args.include_iphone,
    #             max_dslr_images=args.max_dslr_images,
    #         )
    #         if sample is None:
    #             skipped += 1
    #             print(f"[SKIP] {scene_id}: {reason}")
    #             continue
    #         writer.add_sample(sample)
    #         del sample
    #         processed += 1

    # writer.close()

    # print(f"\n{'='*60}")
    # print(f"[All Done] {writer.summary()}")
    # print(f"  This run — Processed: {processed}, Skipped: {skipped}, "
    #       f"Already existed: {already_done}")
    # print(f"{'='*60}")

    # ── Upload to HuggingFace ──
    if args.hf_repo:
        upload_to_hf(args.output_dir, args.hf_repo, private=args.hf_private)


if __name__ == "__main__":
    main()
