#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/to_webdataset_blendedmvs.py" \
    --data-root /apdcephfs_303747097/share_303747097/jingfanchen/data/blendmvs/download \
    --output-dir /apdcephfs_303747097/share_303747097/jingfanchen/data/blendmvs/blendmvs_wds \
    --max-samples-per-shard 20 \
    --num-workers 4
    # --include-masked \
    # --hf-repo cjfcsjt/blendedmvs-wds \
    # --hf-private
