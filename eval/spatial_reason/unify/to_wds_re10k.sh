#!/usr/bin/env bash
set -euo pipefail

python to_webdataset_re10k.py \
    --annotation-root /apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/realestate10k/RealEstate10K \
    --image-root /apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/datasets--mutou0308--RE10K/snapshots/85f1c43d30031e1cf9764eb30f40daa3ec72f6ae \
    --output-dir /apdcephfs_303747097/share_303747097/jingfanchen/data/RealEstate10K/re10k-wds \
    --max-samples-per-shard 50 \
    --max-frames 0 \
    --splits train test \
    --num-workers 8 \
    # --hf-repo cjfcsjt/re10k-wds \
    # --hf-private
