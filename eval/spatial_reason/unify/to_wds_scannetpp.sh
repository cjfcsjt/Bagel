#!/usr/bin/env bash
set -euo pipefail

python to_webdataset_scannetpp.py \
    --data-root /dfs/dataset/gui_dataset/scannetpp/scannetpp \
    --output-dir /dfs/dataset/gui_dataset/scannetpp/scannetpp-wds \
    --max-samples-per-shard 10 \
    --max-dslr-images 0 \
    --include-mesh \
    --num-workers 8 \
    --include-iphone \
    --hf-repo cjfcsjt/scan-wds-netpp \
    # --no-mesh \
    # --hf-private
