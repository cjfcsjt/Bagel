# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# 切换到项目根目录
cd /data/spatial_data/Bagel
export PYTHONPATH="/data/spatial_data/Bagel:$PYTHONPATH"
# 设置wandb配置
export WANDB_API_KEY="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"
export WANDB_ENTITY="jingfan-chen"  # 你的wandb用户名
export WANDB_PROJECT="bagel-training"     # 你可以选择一个项目名

export CUDA_VISIBLE_DEVICES=0
# replace the variables with your own
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=12345 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path /data/spatial_data/hf/flux/ae.safetensors \
  --vit_path /data/spatial_data/hf/siglip-so400m-14-980-flash-attn2-navit \
  --llm_path /data/spatial_data/hf/Qwen2.5-0.5B-Instruct \
  --use_flex True \
  --resume_from None \
  --results_dir results \
  --checkpoint_dir /data/spatial_data/ckpt \
  --max_latent_size 64  \
  --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split