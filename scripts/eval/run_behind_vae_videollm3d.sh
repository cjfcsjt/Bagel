#!/bin/bash
# Behind-VAE multi-view generation evaluation on VideoLLM3D (ScanNet)
# 用法: bash scripts/eval/run_behind_vae_videollm3d.sh
#
# 环境变量覆盖示例:
#   GPUS=4 model_path=/path/to/model dataset=sqa3d num_samples=50 \
#       bash scripts/eval/run_behind_vae_videollm3d.sh

set -x
export HOME=/apdcephfs_303747097/share_303747097/jingfanchen/
cd $HOME/code/Bagel
export PYTHONPATH="$HOME/code/Bagel:$PYTHONPATH"
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
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}

# NCCL settings
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="WARN"
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_HCA="mlx5_0"
export NCCL_P2P_LEVEL="NVL"

GPUS=${GPUS:-8}

# BAGEL 原始模型路径
bagel_base_path=${bagel_base_path:-"/root/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b/"}

# 模型路径（评测用）
model_path=${model_path:-"/tmp/eval_model_behind_vae"}

# 输出路径
output_path=${output_path:-"./results/behind_vae_videollm3d_eval"}

# 自定义模型权重（训练后的 checkpoint），留空则使用 model_path/ema.safetensors
model_weights=${model_weights:-""}

# 数据集选择: scanqa / sqa3d / scan2cap / scanrefer / multi3drefer
dataset=${dataset:-"scanqa"}

# 自定义 JSON 数据文件路径（留空则使用默认路径）
json_path=${json_path:-""}

# VideoProcessor 参数
video_folder=${video_folder:-"/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/"}
annotation_dir=${annotation_dir:-"/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/embodiedscan/"}
metadata_dir=${metadata_dir:-"/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/metadata/"}
frame_sampling_strategy=${frame_sampling_strategy:-"uniform"}
frames_upbound=${frames_upbound:-32}

# ref/nonref 参数（-1 表示随机选择 N//4 或 N//2，与训练一致）
ref_num=${ref_num:--1}

# mask 参数（与训练脚本 train_sft_vae.sh 中的 --mask_mode / --mask_ratio 一致）
mask_mode=${mask_mode:-"random,rectangle,ellipse"}
mask_ratio=${mask_ratio:-"0.9,0.75,0.75"}

# 生成参数
num_samples=${num_samples:-5}
cfg_text_scale=${cfg_text_scale:-4.0}
cfg_img_scale=${cfg_img_scale:-2.0}
cfg_renorm_type=${cfg_renorm_type:-"text_channel"}
cfg_renorm_min=${cfg_renorm_min:-0.0}
timestep_shift=${timestep_shift:-3.0}
num_timesteps=${num_timesteps:-50}

# VAE/VIT transform 参数（与训练时 joint_train.yaml 一致）
vae_max_image_size=${vae_max_image_size:-256}
vae_min_image_size=${vae_min_image_size:-256}
vit_max_image_size=${vit_max_image_size:-384}
vit_min_image_size=${vit_min_image_size:-256}

seed=${seed:-42}

# 构建可选参数
model_weights_arg=""
if [ -n "$model_weights" ]; then
    model_weights_arg="--model_weights $model_weights"
fi

json_path_arg=""
if [ -n "$json_path" ]; then
    json_path_arg="--json_path $json_path"
fi

force_sample_arg=""
if [ "${force_sample:-true}" = "true" ]; then
    force_sample_arg="--force_sample"
fi

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12348 \
    ./eval/gen/gen_images_behind_vae_videollm3d.py \
    --output_dir $output_path \
    --dataset $dataset \
    $json_path_arg \
    --model-path $model_path \
    $model_weights_arg \
    --video_folder $video_folder \
    --annotation_dir $annotation_dir \
    --metadata_dir $metadata_dir \
    --frame_sampling_strategy $frame_sampling_strategy \
    $force_sample_arg \
    --frames_upbound $frames_upbound \
    --ref_num $ref_num \
    --mask_mode "$mask_mode" \
    --mask_ratio "$mask_ratio" \
    --num_samples $num_samples \
    --cfg_text_scale $cfg_text_scale \
    --cfg_img_scale $cfg_img_scale \
    --cfg_renorm_type $cfg_renorm_type \
    --cfg_renorm_min $cfg_renorm_min \
    --timestep_shift $timestep_shift \
    --num_timesteps $num_timesteps \
    --vae_max_image_size $vae_max_image_size \
    --vae_min_image_size $vae_min_image_size \
    --vit_max_image_size $vit_max_image_size \
    --vit_min_image_size $vit_min_image_size \
    --max_latent_size 64 \
    --seed $seed
