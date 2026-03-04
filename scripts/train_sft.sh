# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# 切换到项目根目录
# /data/spatial_data/data/blendedmvs
cd /data/spatial_data/Bagel
export PYTHONPATH="/data/spatial_data/Bagel:$PYTHONPATH"
# 设置wandb配置
export WANDB_API_KEY="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"
export WANDB_ENTITY="jingfan-chen"  # 你的wandb用户名
export WANDB_PROJECT="bagel-sft-training-sh"     # 你可以选择一个项目名
# export NCCL_NET_PLUGIN=none
export CUDA_VISIBLE_DEVICES=0,1,2,3
# replace the variables with your own
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=12345 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/joint_train.yaml \
  --layer_module Qwen2VLMoTDecoderLayer \
  --copy_init_moe True \
  --vae_path /data/spatial_data/hf/flux/ae.safetensors \
  --dino_path /data/spatial_data/hf/dinov2-with-registers-base \
  --vit_path /data/spatial_data/hf/qwen2-vl-2b \
  --llm_path /data/spatial_data/hf/qwen2-vl-2b \
  --vit_type qwen2vl \
  --visual_recon True \
  --visual_und True \
  --joint_train_recon True \
  --pretrain_train_recon False \
  --use_dino_masking True \
  --ssl True \
  --max_num_tokens_per_sample 10240 \
  --max_num_tokens 11520 \
  --expected_num_tokens 10240 \
  --num_shard 4 \
  --use_flex True \
  --resume_from None \
  --results_dir results \
  --checkpoint_dir /data/spatial_data/ckpt \
  --save_every 200 \
  --log_every 1 \
  --lr 2e-5 \
  --max_latent_size 64  \
  --wandb_runid 43 \
  --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split