#!/usr/bin/env python3
"""
Convert BlendedMVS(++) data into **WebDataset** shards and
(optionally) upload them to a HuggingFace Hub dataset repository.

BlendedMVS directory layout
---------------------------
::

    <data-root>/
        <scene_id>/          # e.g. 000000000000000000000000
            blended_images/  # XXXXXXXX.jpg  +  XXXXXXXX_masked.jpg
            cams/            # XXXXXXXX_cam.txt  (extrinsic + intrinsic + depth range)
            rendered_depth_maps/  # XXXXXXXX.pfm  (float32 depth)

Each WebDataset sample corresponds to **one scene** and contains:
    - ``blended_images/{fname}``         : RGB frames (*.jpg) — originals only, masked excluded by default
    - ``blended_images_masked/{fname}``  : masked RGB frames (optional, controlled by --include-masked)
    - ``rendered_depth_maps/{fname}``    : depth maps (*.pfm)
    - ``cams/{fname}``                   : camera parameter files (*.txt)
    - ``meta.json``                      : scene id, number of frames, etc.

Usage
-----
python to_webdataset_blendedmvs.py \\
    --data-root /path/to/blendmvs/download \\
    --output-dir /path/to/wds_output \\
    --max-samples-per-shard 20 \\
    --num-workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  Scene data loader
# ═══════════════════════════════════════════════════════════════════════

def _load_files_from_dir(directory: str, ext: str) -> Dict[str, bytes]:
    """Load all files with given extension from a directory."""
    result = {}
    if not os.path.isdir(directory):
        return result
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(ext):
            fpath = os.path.join(directory, fname)
            with open(fpath, "rb") as f:
                result[fname] = f.read()
    return result


def process_single_scene(
    scene_dir: str,
    scene_id: str,
    include_masked: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Process a single BlendedMVS scene and return a dict ready to be
    written into a WebDataset tar shard.

    Returns None on failure (no images found, etc.).
    """
    images_dir = os.path.join(scene_dir, "blended_images")
    cams_dir = os.path.join(scene_dir, "cams")
    depth_dir = os.path.join(scene_dir, "rendered_depth_maps")

    # ── Load RGB images (exclude *_masked.jpg by default) ──
    all_jpg = _load_files_from_dir(images_dir, ".jpg")
    rgb_files: Dict[str, bytes] = {}
    masked_files: Dict[str, bytes] = {}
    for fname, data in all_jpg.items():
        if "_masked" in fname:
            masked_files[fname] = data
        else:
            rgb_files[fname] = data

    if not rgb_files:
        return None

    rgb_fnames = sorted(rgb_files.keys())
    n_frames = len(rgb_fnames)

    # ── Load camera parameter files ──
    cam_files = _load_files_from_dir(cams_dir, ".txt")

    # ── Load depth maps (.pfm) ──
    depth_files = _load_files_from_dir(depth_dir, ".pfm")

    # ── Build sample dict ──
    sample: Dict[str, Any] = {"__key__": scene_id}

    # RGB frames
    for fname in rgb_fnames:
        sample[f"blended_images/{fname}"] = rgb_files[fname]

    # Masked frames (optional)
    if include_masked:
        for fname in sorted(masked_files.keys()):
            sample[f"blended_images_masked/{fname}"] = masked_files[fname]

    # Camera parameter files
    for fname in sorted(cam_files.keys()):
        sample[f"cams/{fname}"] = cam_files[fname]

    # Depth maps
    for fname in sorted(depth_files.keys()):
        sample[f"rendered_depth_maps/{fname}"] = depth_files[fname]

    # Metadata
    meta = {
        "scene_id": scene_id,
        "n_frames": n_frames,
        "rgb_filenames": rgb_fnames,
        "has_depth": bool(depth_files),
        "has_cams": bool(cam_files),
        "has_masked": bool(masked_files),
        "n_cams": len(cam_files),
        "n_depth": len(depth_files),
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

    def __init__(self, output_dir: str, prefix: str = "blendedmvs",
                 max_samples: int = 20):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_samples = max_samples

        os.makedirs(output_dir, exist_ok=True)
        self.shard_idx = 0
        self.sample_count = 0
        self.total_samples = 0
        self.tar: Optional[tarfile.TarFile] = None
        self.shard_paths: List[str] = []
        self._current_shard_scenes: List[str] = []
        self._shard_scene_map: Dict[str, List[str]] = {}
        self._scene_shard_map: Dict[str, str] = {}
        self._open_new_shard()

    def _open_new_shard(self):
        """Open a new tar shard file."""
        if self.tar is not None:
            self.tar.close()
            prev_shard = os.path.basename(self.shard_paths[-1])
            self._shard_scene_map[prev_shard] = list(self._current_shard_scenes)
            self._current_shard_scenes = []
        shard_name = f"{self.prefix}-{self.shard_idx:06d}.tar"
        shard_path = os.path.join(self.output_dir, shard_name)
        self.tar = tarfile.open(shard_path, "w")
        self.shard_paths.append(shard_path)
        self.sample_count = 0

    def add_sample(self, sample: Dict[str, Any]):
        """Add one sample (scene) to the current shard."""
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
            if self.shard_paths:
                last_shard = os.path.basename(self.shard_paths[-1])
                self._shard_scene_map[last_shard] = list(self._current_shard_scenes)
                self._current_shard_scenes = []

        if self.total_samples == 0:
            return

        # ── Write shards.txt ──
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
            "dataset": "BlendedMVS",
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
    """Upload all shard .tar files to a HuggingFace Hub dataset repo."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"[HF] Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"[HF] Warning creating repo: {e}")

    # Collect all tar/txt/json files to upload
    files_to_upload = []
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if fname.endswith((".tar", ".txt", ".json")):
            files_to_upload.append((fpath, f"data/{fname}"))

    print(f"[HF] Uploading {len(files_to_upload)} file(s) to {repo_id} ...")
    for fpath, remote_path in tqdm(files_to_upload, desc="Uploading"):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload a README
    readme_content = f"""---
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/*.tar
---

# BlendedMVS WebDataset

This dataset contains **BlendedMVS** data converted to
[WebDataset](https://github.com/webdataset/webdataset) format.

Each sample is one multi-view scene with:
- ``blended_images/`` — RGB frames (768×576 JPG)
- ``rendered_depth_maps/`` — rendered depth maps (PFM float32)
- ``cams/`` — camera parameters (extrinsic 4×4 + intrinsic 3×3 + depth range)
- ``meta.json`` — scene metadata
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] Upload complete: https://huggingface.co/datasets/{repo_id}")


# ═══════════════════════════════════════════════════════════════════════
#  Worker function for parallel processing
# ═══════════════════════════════════════════════════════════════════════

def _worker_process_scene(args_tuple):
    """Worker function for ProcessPoolExecutor."""
    scene_dir, scene_id, include_masked = args_tuple
    try:
        return process_single_scene(scene_dir, scene_id, include_masked)
    except Exception as e:
        print(f"[WARN] Failed to process scene {scene_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert BlendedMVS to WebDataset shards and "
                    "optionally upload to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to BlendedMVS download directory containing scene folders",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write WebDataset .tar shards",
    )
    parser.add_argument(
        "--max-samples-per-shard", type=int, default=20,
        help="Maximum number of scenes per .tar shard",
    )
    parser.add_argument(
        "--include-masked", action="store_true", default=False,
        help="Include *_masked.jpg images in the shards",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo ID to upload to "
             "(e.g. 'username/blendedmvs-wds'). "
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

    # ── Discover scenes ──
    # Each sub-directory that contains a "blended_images" folder is a scene
    scene_ids = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and os.path.isdir(os.path.join(data_root, d, "blended_images"))
    )

    print(f"[BlendedMVS → WebDataset]")
    print(f"  Data root      : {data_root}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Scenes found   : {len(scene_ids)}")
    print(f"  Samples/shard  : {args.max_samples_per_shard}")
    print(f"  Include masked : {args.include_masked}")
    print(f"  Workers        : {args.num_workers or 'sequential'}")
    print()

    writer = ShardWriter(
        args.output_dir,
        prefix="blendedmvs",
        max_samples=args.max_samples_per_shard,
    )

    processed = 0
    skipped = 0

    if args.num_workers > 0:
        # ── Parallel processing ──
        tasks = [
            (os.path.join(data_root, sid), sid, args.include_masked)
            for sid in scene_ids
        ]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_worker_process_scene, t): t[1]
                for t in tasks
            }
            pbar = tqdm(
                as_completed(futures), total=len(futures),
                desc="Processing scenes",
            )
            for future in pbar:
                scene_id = futures[future]
                try:
                    sample = future.result()
                except Exception as e:
                    print(f"[WARN] Scene {scene_id} raised: {e}")
                    sample = None

                if sample is None:
                    skipped += 1
                else:
                    writer.add_sample(sample)
                    processed += 1
                pbar.set_postfix(ok=processed, skip=skipped)
    else:
        # ── Sequential processing ──
        for scene_id in tqdm(scene_ids, desc="Processing scenes"):
            scene_dir = os.path.join(data_root, scene_id)
            sample = process_single_scene(
                scene_dir, scene_id, args.include_masked
            )
            if sample is None:
                skipped += 1
                continue
            writer.add_sample(sample)
            processed += 1

    writer.close()

    print(f"\n{'='*60}")
    print(f"[Done] {writer.summary()}")
    print(f"  Processed: {processed}, Skipped: {skipped}")
    print(f"{'='*60}")

    # ── Upload to HuggingFace ──
    if args.hf_repo:
        upload_to_hf(args.output_dir, args.hf_repo, private=args.hf_private)


if __name__ == "__main__":
    main()
