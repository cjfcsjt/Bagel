#!/usr/bin/env python3
"""
Extract ScanNet++ WebDataset .tar shards back to the original directory layout.

Supports **resume**: a progress file tracks which (shard, scene) pairs have
already been fully extracted, so re-running the script after an interruption
will skip completed work.

Usage
-----
python extract_webdataset_scannetpp.py \
    --shard-dir /path/to/wds_output \
    --output-dir /path/to/extracted \
    --num-workers 4

The extracted layout mirrors the original ScanNet++ structure:

    {output-dir}/
        data/
            {scene_id}/
                dslr/
                    resized_undistorted_images/
                    ...
                scans/
                    ...
                meta.json
        splits/
            nvs_sem_train.txt
            ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  Progress tracker (for resume support)
# ═══════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Track which (shard_name, scene_id) pairs have been fully extracted.
    Persists state to a JSON file so extraction can resume after interruption.
    """

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.completed: Dict[str, List[str]] = {}  # shard_name -> [scene_ids]
        self._load()

    def _load(self):
        if os.path.isfile(self.progress_file):
            with open(self.progress_file, "r") as f:
                data = json.load(f)
            self.completed = data.get("completed", {})

    def save(self):
        with open(self.progress_file, "w") as f:
            json.dump({"completed": self.completed}, f, indent=2)

    def is_scene_done(self, shard_name: str, scene_id: str) -> bool:
        return scene_id in self.completed.get(shard_name, [])

    def mark_scene_done(self, shard_name: str, scene_id: str):
        if shard_name not in self.completed:
            self.completed[shard_name] = []
        if scene_id not in self.completed[shard_name]:
            self.completed[shard_name].append(scene_id)

    def is_shard_done(self, shard_name: str, expected_scenes: Optional[List[str]] = None) -> bool:
        """Check if an entire shard has been fully extracted."""
        if expected_scenes is not None:
            done = set(self.completed.get(shard_name, []))
            return all(s in done for s in expected_scenes)
        # If we don't know expected scenes, just check if shard key exists
        return shard_name in self.completed and len(self.completed[shard_name]) > 0

    def total_scenes_done(self) -> int:
        return sum(len(v) for v in self.completed.values())


# ═══════════════════════════════════════════════════════════════════════
#  Extraction logic
# ═══════════════════════════════════════════════════════════════════════

def extract_shard(
    shard_path: str,
    output_dir: str,
    tracker: ProgressTracker,
    expected_scenes: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Extract all scenes from a single .tar shard.

    Each tar member has the name format: ``{scene_id}/{relative_path}``.
    Files are written to ``{output_dir}/data/{scene_id}/{relative_path}``.

    Returns (extracted_count, skipped_count).
    """
    shard_name = os.path.basename(shard_path)
    data_out = os.path.join(output_dir, "data")

    # Group tar members by scene_id
    scene_members: Dict[str, List[tarfile.TarInfo]] = {}

    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # member.name = "{scene_id}/{rest_of_path}"
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            scene_id = parts[0]
            if scene_id not in scene_members:
                scene_members[scene_id] = []
            scene_members[scene_id].append(member)

    extracted = 0
    skipped = 0

    for scene_id in sorted(scene_members.keys()):
        # Check if already extracted
        if tracker.is_scene_done(shard_name, scene_id):
            print(f"  [SKIP] {shard_name} / {scene_id} (already done)", flush=True)
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would extract scene: {scene_id} "
                  f"({len(scene_members[scene_id])} files)", flush=True)
            extracted += 1
            continue

        print(f"  [EXTRACTING] {shard_name} / {scene_id} "
              f"({len(scene_members[scene_id])} files) ...", flush=True)

        # Extract all files for this scene
        scene_out_dir = os.path.join(data_out, scene_id)
        with tarfile.open(shard_path, "r") as tar:
            for member in scene_members[scene_id]:
                # member.name = "{scene_id}/{rel_path}"
                parts = member.name.split("/", 1)
                rel_path = parts[1]

                out_path = os.path.join(scene_out_dir, rel_path)
                out_parent = os.path.dirname(out_path)
                os.makedirs(out_parent, exist_ok=True)

                # Extract file content
                f = tar.extractfile(member)
                if f is None:
                    continue
                with open(out_path, "wb") as out_f:
                    # Read in chunks to handle large files
                    while True:
                        chunk = f.read(8 * 1024 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        out_f.write(chunk)
                f.close()

        # Mark as done and persist progress
        tracker.mark_scene_done(shard_name, scene_id)
        tracker.save()
        extracted += 1
        print(f"  [DONE] {shard_name} / {scene_id} extracted successfully", flush=True)

    return extracted, skipped


def extract_split_files(shard_dir: str, output_dir: str):
    """
    Copy split .txt files from the shard directory to {output_dir}/splits/.
    These were placed alongside the .tar shards during packing.
    """
    splits_out = os.path.join(output_dir, "splits")
    copied = 0
    for fname in sorted(os.listdir(shard_dir)):
        if fname.endswith(".txt") and fname != "shards.txt":
            src = os.path.join(shard_dir, fname)
            if not os.path.isfile(src):
                continue
            os.makedirs(splits_out, exist_ok=True)
            dst = os.path.join(splits_out, fname)
            if not os.path.isfile(dst):
                import shutil
                shutil.copy2(src, dst)
                copied += 1
                print(f"  Copied split file: {fname}")
    if copied == 0:
        print("  No new split files to copy.")


# ═══════════════════════════════════════════════════════════════════════
#  Parallel worker
# ═══════════════════════════════════════════════════════════════════════

def _extract_worker(args_tuple):
    """Worker function for parallel extraction of a single shard."""
    shard_path, output_dir, progress_file, expected_scenes = args_tuple
    tracker = ProgressTracker(progress_file)
    try:
        extracted, skipped = extract_shard(
            shard_path, output_dir, tracker, expected_scenes
        )
        return shard_path, extracted, skipped, None
    except Exception as e:
        return shard_path, 0, 0, str(e)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract ScanNet++ WebDataset .tar shards back to "
                    "the original directory layout (with resume support).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shard-dir", type=str, required=True,
        help="Directory containing the .tar shards, metadata.json, "
             "shards.txt, and split .txt files.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to extract scenes into. Scenes will be placed "
             "under {output-dir}/data/{scene_id}/...",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Only print what would be extracted, without writing files.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of parallel workers (0 = sequential). "
             "Each worker handles one shard at a time.",
    )
    parser.add_argument(
        "--scenes", type=str, nargs="*", default=None,
        help="Only extract these specific scene IDs (space-separated). "
             "If not specified, extract all scenes.",
    )
    args = parser.parse_args()

    shard_dir = args.shard_dir
    output_dir = args.output_dir

    if not os.path.isdir(shard_dir):
        print(f"[ERROR] Shard directory not found: {shard_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load metadata if available ──
    metadata_path = os.path.join(shard_dir, "metadata.json")
    metadata = None
    shard_scene_map: Dict[str, List[str]] = {}

    if os.path.isfile(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for shard_info in metadata.get("shards", []):
            shard_scene_map[shard_info["shard_name"]] = shard_info.get("scenes", [])
        print(f"[Metadata] Loaded: {metadata.get('total_scenes', '?')} scenes "
              f"across {metadata.get('total_shards', '?')} shard(s)")
    else:
        print("[WARN] No metadata.json found; will discover .tar files directly.")

    # ── Discover shard files ──
    if shard_scene_map:
        shard_files = [
            os.path.join(shard_dir, name)
            for name in sorted(shard_scene_map.keys())
            if os.path.isfile(os.path.join(shard_dir, name))
        ]
    else:
        shard_files = sorted(
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.endswith(".tar")
        )

    if not shard_files:
        print(f"[ERROR] No .tar shard files found in {shard_dir}")
        sys.exit(1)

    # ── Filter shards if specific scenes requested ──
    target_scenes = set(args.scenes) if args.scenes else None
    if target_scenes and shard_scene_map:
        # Only process shards that contain requested scenes
        filtered = []
        for sf in shard_files:
            sname = os.path.basename(sf)
            scenes_in_shard = shard_scene_map.get(sname, [])
            if any(s in target_scenes for s in scenes_in_shard):
                filtered.append(sf)
        shard_files = filtered
        print(f"[Filter] Processing {len(shard_files)} shard(s) "
              f"containing requested scenes: {sorted(target_scenes)}")

    # ── Progress tracker ──
    progress_file = os.path.join(output_dir, ".extraction_progress.json")
    tracker = ProgressTracker(progress_file)

    already_done = tracker.total_scenes_done()
    if already_done > 0:
        print(f"[Resume] Found progress file: {already_done} scene(s) "
              f"already extracted.")

    # ── Copy split files ──
    print("\n── Copying split files ──")
    extract_split_files(shard_dir, output_dir)

    # ── Extract shards ──
    print(f"\n── Extracting {len(shard_files)} shard(s) ──")

    total_extracted = 0
    total_skipped = 0
    t0 = time.time()

    if args.num_workers > 0 and len(shard_files) > 1:
        # Parallel extraction using ThreadPoolExecutor at scene level.
        # ThreadPoolExecutor ensures print() in workers is visible in the
        # main terminal (unlike ProcessPoolExecutor which runs in subprocesses).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Thread-safe lock for progress tracker and progress bar
        _lock = threading.Lock()

        # ── Step 1: Build a list of (shard_path, scene_id) tasks ──
        # We need to peek into each shard to discover scenes.
        print("[Info] Scanning shards to build scene task list ...", flush=True)
        scene_tasks: List[Tuple[str, str]] = []  # (shard_path, scene_id)

        for sf in shard_files:
            sname = os.path.basename(sf)
            scenes_in_shard = shard_scene_map.get(sname, [])
            if scenes_in_shard:
                for sid in scenes_in_shard:
                    scene_tasks.append((sf, sid))
            else:
                # No metadata; peek into tar to discover scene IDs
                try:
                    with tarfile.open(sf, "r") as tar:
                        scene_ids_found = set()
                        for member in tar.getmembers():
                            if member.isfile():
                                parts = member.name.split("/", 1)
                                if len(parts) >= 2:
                                    scene_ids_found.add(parts[0])
                        for sid in sorted(scene_ids_found):
                            scene_tasks.append((sf, sid))
                except Exception as e:
                    print(f"[WARN] Failed to peek {sname}: {e}", flush=True)

        total_scene_count = len(scene_tasks)
        print(f"[Info] Total scenes to process: {total_scene_count}", flush=True)

        # ── Step 2: Define per-scene worker function ──
        def _extract_single_scene(shard_path: str, scene_id: str):
            """Extract a single scene from a shard. Returns (scene_id, status, error)."""
            shard_name = os.path.basename(shard_path)
            data_out = os.path.join(output_dir, "data")

            # Check if already done (thread-safe read)
            with _lock:
                if tracker.is_scene_done(shard_name, scene_id):
                    return scene_id, "skipped", None

            print(f"  [EXTRACTING] {shard_name} / {scene_id} ...", flush=True)

            try:
                # Open tar and extract only this scene's files
                scene_out_dir = os.path.join(data_out, scene_id)
                file_count = 0
                with tarfile.open(shard_path, "r") as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        parts = member.name.split("/", 1)
                        if len(parts) < 2:
                            continue
                        if parts[0] != scene_id:
                            continue

                        rel_path = parts[1]
                        out_path = os.path.join(scene_out_dir, rel_path)
                        out_parent = os.path.dirname(out_path)
                        os.makedirs(out_parent, exist_ok=True)

                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        with open(out_path, "wb") as out_f:
                            while True:
                                chunk = f.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                out_f.write(chunk)
                        f.close()
                        file_count += 1

                # Mark done (thread-safe write)
                with _lock:
                    tracker.mark_scene_done(shard_name, scene_id)
                    tracker.save()

                print(f"  [DONE] {shard_name} / {scene_id} ({file_count} files)", flush=True)
                return scene_id, "extracted", None

            except Exception as e:
                return scene_id, "error", str(e)

        # ── Step 3: Submit all scene tasks to thread pool ──
        pbar = tqdm(total=total_scene_count, desc="Extracting scenes", unit="scene")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_extract_single_scene, sp, sid): (sp, sid)
                for sp, sid in scene_tasks
            }
            for future in as_completed(futures):
                sp, sid = futures[future]
                shard_name = os.path.basename(sp)
                try:
                    _, status, error = future.result()
                    if status == "extracted":
                        total_extracted += 1
                    elif status == "skipped":
                        total_skipped += 1
                    elif status == "error":
                        print(f"\n[ERROR] {shard_name} / {sid}: {error}", flush=True)
                except Exception as e:
                    print(f"\n[ERROR] {shard_name} / {sid}: {e}", flush=True)
                pbar.update(1)
                pbar.set_postfix(extracted=total_extracted, skipped=total_skipped)
            pbar.close()
    else:
        # Sequential extraction with scene-level progress bar
        # First, count total scenes across all shards
        total_scene_count = 0
        for shard_path in shard_files:
            shard_name = os.path.basename(shard_path)
            expected = shard_scene_map.get(shard_name, None)
            if expected:
                total_scene_count += len(expected)
            else:
                # Peek into tar to count scenes
                print(f"[Info] Peeking into {shard_name} to count scenes ...", flush=True)
                try:
                    with tarfile.open(shard_path, "r") as tar:
                        scene_ids = set()
                        for member in tar.getmembers():
                            if member.isfile():
                                parts = member.name.split("/", 1)
                                if len(parts) >= 2:
                                    scene_ids.add(parts[0])
                        total_scene_count += len(scene_ids)
                except Exception:
                    total_scene_count += 1  # fallback

        print(f"[Info] Total scenes to process: {total_scene_count}", flush=True)
        pbar = tqdm(total=total_scene_count, desc="Extracting scenes", unit="scene")

        for shard_path in shard_files:
            shard_name = os.path.basename(shard_path)
            expected = shard_scene_map.get(shard_name, None)

            print(f"\n[Shard] Processing {shard_name} ...", flush=True)

            # Quick check: skip entire shard if all scenes done
            if expected and tracker.is_shard_done(shard_name, expected):
                print(f"[Shard] {shard_name} fully done, skipping {len(expected)} scene(s)", flush=True)
                total_skipped += len(expected)
                pbar.update(len(expected))
                continue

            extracted, skipped = extract_shard(
                shard_path, output_dir, tracker, expected,
                dry_run=args.dry_run,
            )
            total_extracted += extracted
            total_skipped += skipped
            pbar.update(extracted + skipped)
            pbar.set_postfix(
                shard=shard_name,
                extracted=total_extracted,
                skipped=total_skipped,
            )
            print(f"[Shard Done] {shard_name}: extracted={extracted}, skipped={skipped}", flush=True)

        pbar.close()

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"[Done] Extraction complete in {elapsed:.1f}s")
    print(f"  Extracted : {total_extracted} scene(s)")
    print(f"  Skipped   : {total_skipped} scene(s) (already done)")
    print(f"  Output    : {output_dir}")
    print(f"  Progress  : {progress_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
