# SPDX-License-Identifier: Apache-2.0
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
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NCCL settings
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="WARN"
export NCCL_BLOCKING_WAIT=1
# export NCCL_SOCKET_IFNAME="bond0"
# export NCCL_NET_PLUGIN=none
export NCCL_IB_HCA="mlx5_0"
export NCCL_P2P_LEVEL="NVL"
RUNID=102
# replace the variables with your own
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=12345 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/joint_train.yaml \
  --layer_module Qwen2VLMoTDecoderLayer \
  --copy_init_moe True \
  --vae_path $HOME/hf/flux/ae.safetensors \
  --dino_path facebook/dinov2-with-registers-base \
  --vit_path Qwen/Qwen2-VL-2B-Instruct \
  --llm_path Qwen/Qwen2-VL-2B-Instruct \
  --vit_type qwen2vl \
  --visual_recon True \
  --visual_und True \
  --joint_train_recon True \
  --pretrain_train_recon False \
  --ssl True \
  --use_dino_masking True \
  --dino_mask_mode "random,rectangle,ellipse" \
  --dino_mask_ratio "0.9,0.75,0.75" \
  --dino_num_ref -1 \
  --max_num_tokens 50000 \
  --expected_num_tokens 50000 \
  --max_num_tokens_per_sample 25000 \
  --num_shard 8 \
  --cpu_offload False \
  --use_flex True \
  --resume_from None \
  --results_dir ./results \
  --checkpoint_dir $HOME/code/ckpt/joint_geo_then_und_dual_${RUNID} \
  --save_every 200 \
  --log_every 1 \
  --lr 2e-5 \
  --max_latent_size 64  \
  --wandb_runid ${RUNID} \
  --num_workers 2 # use small num_workers since the num_used_data (10) are not enough to split