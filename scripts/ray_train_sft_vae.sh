# SPDX-License-Identifier: Apache-2.0
# =====================================================================
# Ray Train 分布式训练启动脚本
# 用法: bash scripts/ray_train_sft_vae.sh
# =====================================================================

# Data path settings
export HOME=/apdcephfs_303747097/share_303747097/jingfanchen/
cd $HOME/code/Bagel
export PYTHONPATH="$HOME/code/Bagel:$PYTHONPATH"
export HF_HOME=$HOME/.cache/huggingface

# 设置wandb配置
export WANDB_API_KEY="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"
export WANDB_ENTITY="jingfan-chen"  # 你的wandb用户名
export WANDB_PROJECT="bagel-sft-training-sh"     # 你可以选择一个项目名

# Thread settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# CUDA settings
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL settings
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="WARN"
export NCCL_BLOCKING_WAIT=1
# export NCCL_SOCKET_IFNAME="bond0"
# export NCCL_NET_PLUGIN=none
export NCCL_IB_HCA="mlx5_0"
export NCCL_P2P_LEVEL="NVL"

RUNID=145

# =====================================================================
# 使用 Ray Train 启动（替代 torchrun）
# --ray_address: 设为 "auto" 连接已有集群，或留空在本地启动
# --num_ray_workers: 总 GPU worker 数（单机8卡=8，双机8卡=16）
# =====================================================================
python train/ray_pretrain_unified_navit_vae.py \
  --ray_address auto \
  --num_ray_workers 16 \
  --num_ray_gpus_per_worker 1 \
  --num_ray_cpus_per_worker 8 \
  --dataset_config_file ./data/configs/joint_train.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path /root/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b/ \
  --resume-from /root/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b/ \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --visual_gen False \
  --visual_und True \
  --use_masking False \
  --use_mae_masking False \
  --mask_mode "random,rectangle,ellipse" \
  --mask_ratio "0.9,0.75,0.75" \
  --max_num_tokens 52000 \
  --expected_num_tokens 50000 \
  --max_num_tokens_per_sample 45000 \
  --num_shard 16 \
  --cpu_offload False \
  --use_flex True \
  --results_dir ./results \
  --checkpoint_dir $HOME/code/ckpt/joint_vae_geo_und_mindcube10k_raw_qa_shuffle_64group_${RUNID} \
  --save_every 200 \
  --log_every 1 \
  --gradient_accumulation_steps 2 \
  --lr 2e-5 \
  --max_latent_size 64  \  
  --wandb_runid ${RUNID} \
  --num_workers 2 # use small num_workers since the num_used_data (10) are not enough to split