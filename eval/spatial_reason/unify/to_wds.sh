#!/usr/bin/env bash
set -euo pipefail

python to_webdataset.py \
    --data-root /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/ar_raw_all/raw \
    --output-dir /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ark-wds-itscenes \
    --max-samples-per-shard 50 \
    --num-workers -1 \
    --hf-repo cjfcsjt/ark-wds-itscenes \
    # --hf-private
