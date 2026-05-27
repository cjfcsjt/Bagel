#!/bin/bash
# Behind-VAE multi-view generation evaluation on RE10K
# 用法: bash scripts/eval/run_behind_vae_re10k.sh
#
# 环境变量覆盖示例:
#   GPUS=4 model_path=/path/to/model model_weights=/path/to/ckpt.safetensors \
#       num_scenes=50 bash scripts/eval/run_behind_vae_re10k.sh

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

GPUS=${GPUS:-8}

# BAGEL 原始模型路径（包含配置文件、tokenizer、ae.safetensors 等）
bagel_base_path=${bagel_base_path:-"/root/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b/"}

# 模型路径（评测用，指向 /tmp 临时目录，需先运行 prepare_eval_model.sh 准备文件）
# 如需使用其他路径，可通过环境变量覆盖: model_path=/path/to/model bash scripts/eval/run_behind_vae_re10k.sh
model_path=${model_path:-"/tmp/eval_model_behind_vae"}

# ── 自动补全配置文件：从 bagel_base_path 复制缺失的 *.json, *.txt, ae.safetensors 到 model_path ──
# if [ "$model_path" != "$bagel_base_path" ]; then
#     echo ">>> 检查并补全配置文件: $bagel_base_path -> $model_path"
#     for f in config.json llm_config.json vit_config.json generation_config.json preprocessor_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt ae.safetensors; do
#         if [ ! -f "$model_path/$f" ] && [ -f "$bagel_base_path/$f" ]; then
#             echo "  复制 $f"
#             cp "$bagel_base_path/$f" "$model_path/$f"
#         fi
#     done
#     echo ">>> 配置文件补全完成"
# fi

# 输出路径
output_path=${output_path:-"./results/behind_vae_re10k_eval"}

# 自定义模型权重（训练后的 checkpoint），留空则使用 model_path/ema.safetensors
model_weights=${model_weights:-""}

# RE10K parquet 文件
parquet_file=${parquet_file:-"/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/re10k_00000.parquet"}

# RE10K 图片根目录（留空则使用脚本内默认值）
image_root=${image_root:-""}

# 生成参数
ref_num=${ref_num:-2}
max_frames=${max_frames:-8}
num_scenes=${num_scenes:-5}
cfg_text_scale=${cfg_text_scale:-4.0}
cfg_img_scale=${cfg_img_scale:-2.0}
cfg_renorm_type=${cfg_renorm_type:-"text_channel"}
cfg_renorm_min=${cfg_renorm_min:-0.0}
timestep_shift=${timestep_shift:-3.0}
num_timesteps=${num_timesteps:-50}
max_image_size=${max_image_size:-1024}
min_image_size=${min_image_size:-512}
seed=${seed:-42}

# 构建可选参数
model_weights_arg=""
if [ -n "$model_weights" ]; then
    model_weights_arg="--model_weights $model_weights"
fi

image_root_arg=""
if [ -n "$image_root" ]; then
    image_root_arg="--image_root $image_root"
fi

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12347 \
    ./eval/gen/gen_images_behind_vae_re10k.py \
    --output_dir $output_path \
    --parquet_file $parquet_file \
    $image_root_arg \
    --model-path $model_path \
    $model_weights_arg \
    --ref_num $ref_num \
    --max_frames $max_frames \
    --num_scenes $num_scenes \
    --cfg_text_scale $cfg_text_scale \
    --cfg_img_scale $cfg_img_scale \
    --cfg_renorm_type $cfg_renorm_type \
    --cfg_renorm_min $cfg_renorm_min \
    --timestep_shift $timestep_shift \
    --num_timesteps $num_timesteps \
    --max_image_size $max_image_size \
    --min_image_size $min_image_size \
    --max_latent_size 64 \
    --seed $seed
