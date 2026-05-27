#!/bin/bash
# MAE-style multi-view generation evaluation on RE10K
# 用法: bash scripts/eval/run_mae_re10k.sh

set -x

GPUS=${GPUS:-8}

# BAGEL 原始模型路径（包含配置文件、tokenizer、ae.safetensors 等）
bagel_base_path=${bagel_base_path:-"/root/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b/"}

# 模型路径（评测用，可以是训练后的 checkpoint 目录，只需包含 ema.safetensors）
model_path=${model_path:-"$bagel_base_path"}

# ── 自动补全配置文件：从 bagel_base_path 复制缺失的 *.json, *.txt, ae.safetensors 到 model_path ──
if [ "$model_path" != "$bagel_base_path" ]; then
    echo ">>> 检查并补全配置文件: $bagel_base_path -> $model_path"
    for f in config.json llm_config.json vit_config.json generation_config.json preprocessor_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt ae.safetensors; do
        if [ ! -f "$model_path/$f" ] && [ -f "$bagel_base_path/$f" ]; then
            echo "  复制 $f"
            cp "$bagel_base_path/$f" "$model_path/$f"
        fi
    done
    echo ">>> 配置文件补全完成"
fi

# 输出路径
output_path=${output_path:-"./results/mae_re10k_eval"}

# 自定义模型权重（训练后的 checkpoint），留空则使用 model_path/ema.safetensors
model_weights=${model_weights:-""}

# RE10K parquet 文件
parquet_file=${parquet_file:-"/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/re10k_00000.parquet"}

# 生成参数
ref_num=${ref_num:-2}
max_frames=${max_frames:-8}
num_scenes=${num_scenes:-100}
cfg_text_scale=${cfg_text_scale:-4.0}
cfg_img_scale=${cfg_img_scale:-2.0}
cfg_renorm_type=${cfg_renorm_type:-"text_channel"}
num_timesteps=${num_timesteps:-50}

# 构建 model_weights 参数
model_weights_arg=""
if [ -n "$model_weights" ]; then
    model_weights_arg="--model_weights $model_weights"
fi

# 是否启用 use_mae_masking（加载含 vit_mask_placeholder 的权重时需要）
use_mae_masking_arg=""
if [ "${use_mae_masking:-false}" = "true" ]; then
    use_mae_masking_arg="--use_mae_masking"
fi

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    ./eval/gen/gen_images_mae_re10k.py \
    --output_dir $output_path \
    --parquet_file $parquet_file \
    --model-path $model_path \
    $model_weights_arg \
    --ref_num $ref_num \
    --max_frames $max_frames \
    --num_scenes $num_scenes \
    --cfg_text_scale $cfg_text_scale \
    --cfg_img_scale $cfg_img_scale \
    --cfg_renorm_type $cfg_renorm_type \
    --num_timesteps $num_timesteps \
    --max_latent_size 64 \
    $use_mae_masking_arg
