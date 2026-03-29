#!/usr/bin/env python3
"""
Convert ScanNet (v2) data into **WebDataset** shards and
(optionally) upload them to a HuggingFace Hub dataset repository.

Features
--------
* Reads the official ScanNet directory layout at
  ``{data-root}/data/scans/{scene_id}/`` (train) and
  ``{data-root}/data/scans_test/{scene_id}/`` (test).
* Each WebDataset sample corresponds to **one scene** and contains
  selected files from the scene directory.
* ``.sens`` files are **always skipped** (too large, 0.5–3.5 GB each).
* ``.zip`` files (2D labels/instances) are controlled by
  ``--include-2d-zips`` (default: skip).
* 3D mesh ``.ply`` files are controlled by ``--include-mesh`` / ``--no-mesh``.
* Supports **incremental mode**: re-running with the same ``--output-dir``
  will only process new / previously-skipped scenes.
* Shards are written as ``.tar`` files with configurable max samples per shard.

Usage
-----
python to_webdataset_scannet.py \\
    --data-root /dfs/dataset/gui_dataset/scannet \\
    --output-dir /path/to/wds_output \\
    --max-samples-per-shard 50 \\
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
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Extensions to always skip (too large for shards)
_ALWAYS_SKIP_EXTENSIONS: Set[str] = {} # {".sens"}


# ═══════════════════════════════════════════════════════════════════════
#  Scene data loader
# ═══════════════════════════════════════════════════════════════════════

def process_single_scene(
    scene_dir: str,
    scene_id: str,
    include_mesh: bool = True,
    include_2d_zips: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Process a single ScanNet scene.

    Reads files from the scene directory and returns a dict ready to be
    written into a WebDataset tar shard.

    Skipping rules:
      - ``.sens`` files are always skipped (too large).
      - ``.zip`` files (2D labels / instance masks) are only included
        when *include_2d_zips* is True.
      - 3D mesh ``.ply`` files are only included when *include_mesh* is True.

    Returns
    -------
    (sample_dict, None) on success.
    (None, reason_string) when the scene is skipped.
    """
    if not os.path.isdir(scene_dir):
        return None, f"scene directory does not exist: {scene_dir}"

    sample: Dict[str, Any] = {"__key__": scene_id}
    file_count = 0

    for root, dirs, files in os.walk(scene_dir):
        rel_root = os.path.relpath(root, scene_dir)
        if rel_root == ".":
            rel_root = ""

        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            # Always skip .sens (depth stream, ~0.5-3.5 GB)
            if ext in _ALWAYS_SKIP_EXTENSIONS:
                continue

            # Optionally skip .zip (2D label/instance archives)
            if ext == ".zip" and not include_2d_zips:
                continue

            # Optionally skip mesh .ply files
            if ext == ".ply" and not include_mesh:
                continue

            rel_path = os.path.join(rel_root, fname) if rel_root else fname

            with open(fpath, "rb") as f:
                sample[rel_path] = f.read()

            file_count += 1

    if file_count == 0:
        return None, "no files found in scene directory (all filtered out)"

    # ── Collect file-level metadata ──
    file_list = sorted(k for k in sample if k != "__key__")
    extensions = {}
    for k in file_list:
        ext = os.path.splitext(k)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1

    meta = {
        "scene_id": scene_id,
        "n_files": file_count,
        "file_extensions": extensions,
        "files": file_list,
    }

    # ── Try to parse the scene .txt for extra info ──
    txt_key = f"{scene_id}.txt"
    if txt_key in sample:
        try:
            txt_content = sample[txt_key].decode("utf-8", errors="replace")
            info = {}
            for line in txt_content.strip().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
            if "numColorFrames" in info:
                meta["numColorFrames"] = int(info["numColorFrames"])
            if "numDepthFrames" in info:
                meta["numDepthFrames"] = int(info["numDepthFrames"])
            if "sceneType" in info:
                meta["sceneType"] = info["sceneType"]
        except Exception:
            pass

    sample["meta.json"] = json.dumps(meta).encode("utf-8")

    return sample, None


# ═══════════════════════════════════════════════════════════════════════
#  WebDataset shard writer  (with incremental support)
# ═══════════════════════════════════════════════════════════════════════

class ShardWriter:
    """
    Write samples to WebDataset-style tar shards.

    Each shard is named ``{prefix}-{shard_idx:06d}.tar`` and contains
    at most ``max_samples`` samples.

    Supports **incremental mode**: if a ``metadata.json`` already exists
    in *output_dir*, the writer preserves existing shards and appends
    new ones.
    """

    def __init__(self, output_dir: str, prefix: str = "scannet",
                 max_samples: int = 50):
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
            for shard_info in prev_meta.get("shards", []):
                shard_name = shard_info["shard_name"]
                shard_path = os.path.join(output_dir, shard_name)
                if os.path.isfile(shard_path):
                    self._existing_shard_paths.append(shard_path)
                    self._existing_shard_scene_map[shard_name] = shard_info.get("scenes", [])
            self._existing_scene_shard_map = dict(prev_meta.get("scene_to_shard", {}))
            self._existing_total_samples = prev_meta.get("total_scenes", 0)
            next_idx = len(self._existing_shard_paths)
            print(f"[Incremental] Found existing metadata with "
                  f"{self._existing_total_samples} scenes in "
                  f"{len(self._existing_shard_paths)} shard(s). "
                  f"New shards will start at index {next_idx}.")
        else:
            next_idx = 0

        self.shard_idx = next_idx
        self.sample_count = 0
        self.total_samples = 0  # only NEW samples
        self.tar: Optional[tarfile.TarFile] = None
        self.shard_paths: List[str] = []  # only NEW shard paths
        self._current_shard_scenes: List[str] = []
        self._shard_scene_map: Dict[str, List[str]] = {}
        self._scene_shard_map: Dict[str, str] = {}
        self._new_shard_opened = False

    def get_existing_scene_ids(self) -> set:
        """Return scene IDs already present in existing shards."""
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
        """Add one sample (scene) to the current shard."""
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
        """Close the current shard and write merged metadata."""
        if self.tar is not None:
            self.tar.close()
            self.tar = None
            if self.shard_paths:
                last_shard = os.path.basename(self.shard_paths[-1])
                self._shard_scene_map[last_shard] = list(self._current_shard_scenes)
                self._current_shard_scenes = []

        # ── Merge existing + new ──
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
        path: "scannet-*.tar"
---

# ScanNet v2 WebDataset

This dataset contains **ScanNet v2** data converted to
[WebDataset](https://github.com/webdataset/webdataset) format.

All scenes are stored in flat `.tar` shards.

Each sample is one indoor scene with (depending on options):
- ``{scene_id}.txt``                       — scene metadata (intrinsics, frame counts, etc.)
- ``{scene_id}.aggregation.json``          — object aggregation annotations
- ``{scene_id}_vh_clean.aggregation.json`` — aggregation annotations (clean mesh)
- ``{scene_id}_vh_clean.ply``              — reconstructed mesh
- ``{scene_id}_vh_clean_2.ply``            — decimated mesh
- ``{scene_id}_vh_clean_2.labels.ply``     — semantic-labeled mesh
- ``{scene_id}_vh_clean.segs.json``        — over-segmentation
- ``{scene_id}_vh_clean_2.0.010000.segs.json`` — over-segmentation (decimated)
- ``{scene_id}_2d-label*.zip``             — 2D label frames (optional)
- ``{scene_id}_2d-instance*.zip``          — 2D instance frames (optional)
- ``meta.json``                            — per-sample metadata

Note: ``.sens`` files (raw depth streams) are **not** included due to
their large size (0.5–3.5 GB each).
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
#  Worker for multiprocessing
# ═══════════════════════════════════════════════════════════════════════

def _worker(args_tuple):
    sd, sid, im, iz = args_tuple
    try:
        return process_single_scene(sd, sid, im, iz)
    except Exception as e:
        print(f"[WARN] Failed to process scene {sid}: {e}")
        return None, f"exception: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert ScanNet v2 to WebDataset shards and "
                    "optionally upload to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str,
        default="/dfs/dataset/gui_dataset/scannet",
        help="Path to the ScanNet root directory "
             "(containing data/scans/, data/scans_test/)",
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
        "--include-mesh", action="store_true", default=True,
        help="Include 3D mesh .ply files",
    )
    parser.add_argument(
        "--no-mesh", action="store_true", default=False,
        help="Exclude 3D mesh .ply files to reduce shard size",
    )
    parser.add_argument(
        "--include-2d-zips", action="store_true", default=False,
        help="Include 2D label/instance .zip archives "
             "(can be 50–100 MB each)",
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
        help="Number of parallel workers (0 = sequential).",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)

    include_mesh = args.include_mesh and not args.no_mesh

    # ── Discover ALL scene directories (no split filtering) ──
    scene_entries: List[Tuple[str, str]] = []  # (scene_dir, scene_id)
    train_scene_ids: List[str] = []
    test_scene_ids: List[str] = []

    scans_dir = os.path.join(data_root, "data", "scans")
    scans_test_dir = os.path.join(data_root, "data", "scans_test")

    if os.path.isdir(scans_dir):
        for d in sorted(os.listdir(scans_dir)):
            dp = os.path.join(scans_dir, d)
            if os.path.isdir(dp):
                scene_entries.append((dp, d))
                train_scene_ids.append(d)
        print(f"  Found {len(train_scene_ids)} train scenes in {scans_dir}")
    else:
        print(f"[WARN] Train scans directory not found: {scans_dir}")

    if os.path.isdir(scans_test_dir):
        for d in sorted(os.listdir(scans_test_dir)):
            dp = os.path.join(scans_test_dir, d)
            if os.path.isdir(dp):
                scene_entries.append((dp, d))
                test_scene_ids.append(d)
        print(f"  Found {len(test_scene_ids)} test scenes in {scans_test_dir}")
    else:
        print(f"[WARN] Test scans directory not found: {scans_test_dir}")

    if not scene_entries:
        print(f"[ERROR] No scene directories found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Write split files to output directory ──
    if train_scene_ids:
        split_path = os.path.join(args.output_dir, "scannet_train.txt")
        with open(split_path, "w") as f:
            f.write("\n".join(train_scene_ids) + "\n")
        print(f"  Wrote split file: scannet_train.txt ({len(train_scene_ids)} scenes)")
    if test_scene_ids:
        split_path = os.path.join(args.output_dir, "scannet_test.txt")
        with open(split_path, "w") as f:
            f.write("\n".join(test_scene_ids) + "\n")
        print(f"  Wrote split file: scannet_test.txt ({len(test_scene_ids)} scenes)")

    # ── Copy the labels file if it exists ──
    labels_file = os.path.join(data_root, "data", "scannetv2-labels.combined.tsv")
    if os.path.isfile(labels_file):
        dst = os.path.join(args.output_dir, "scannetv2-labels.combined.tsv")
        shutil.copy2(labels_file, dst)
        print(f"  Copied labels file: scannetv2-labels.combined.tsv")

    all_scene_ids = [sid for _, sid in scene_entries]
    scene_dir_map = {sid: sd for sd, sid in scene_entries}

    print(f"\n[ScanNet → WebDataset]")
    print(f"  Data root      : {data_root}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Total scenes   : {len(all_scene_ids)} "
          f"(train: {len(train_scene_ids)}, test: {len(test_scene_ids)})")
    print(f"  Samples/shard  : {args.max_samples_per_shard}")
    print(f"  Include mesh   : {include_mesh}")
    print(f"  Include 2D zips: {args.include_2d_zips}")
    print(f"  Workers        : {args.num_workers or 'sequential'}")
    print()

    writer = ShardWriter(
        args.output_dir,
        prefix="scannet",
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
    #     from concurrent.futures import ProcessPoolExecutor
    #     import itertools
    #     import concurrent.futures

    #     tasks = [
    #         (scene_dir_map[sid], sid, include_mesh, args.include_2d_zips)
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
    #         scene_dir = scene_dir_map[scene_id]
    #         sample, reason = process_single_scene(
    #             scene_dir, scene_id,
    #             include_mesh=include_mesh,
    #             include_2d_zips=args.include_2d_zips,
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
