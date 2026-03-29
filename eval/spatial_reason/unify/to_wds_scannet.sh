#!/usr/bin/env bash
set -euo pipefail

python to_webdataset_scannet.py \
    --data-root /dfs/dataset/gui_dataset/scannet \
    --output-dir /dfs/dataset/gui_dataset/scannet/scannet-wds \
    --max-samples-per-shard 50 \
    --include-mesh \
    --num-workers 8 \
    --hf-repo cjfcsjt/scan-wds-net \
    # --include-2d-zips \
    # --no-mesh \
    # --hf-private
