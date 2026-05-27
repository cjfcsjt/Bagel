#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# 评测前模型准备脚本
# 功能：将训练权重 + 原始 BAGEL 配置文件复制到 /tmp 临时目录
#
# 用法:
#   bash scripts/eval/prepare_eval_model.sh
#
# 环境变量覆盖示例:
#   ckpt_dir=/path/to/ckpt bagel_base_path=/path/to/bagel \
#       bash scripts/eval/prepare_eval_model.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── 配置参数 ──

# 训练得到的 checkpoint 目录（包含 ema.safetensors 和/或 model.safetensors）
ckpt_dir=${ckpt_dir:-"/apdcephfs_303747097/share_303747097/jingfanchen/code/ckpt/joint_vae_geo_videollm3d_partial_masking_denoise_behind_vae_164/0000800"}

# 原始 BAGEL 模型路径（包含配置文件、tokenizer、ae.safetensors 等）
bagel_base_path=${bagel_base_path:-"/root/.cache/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b/"}

# 目标临时目录
target_dir=${target_dir:-"/tmp/eval_model_behind_vae"}

echo "═══════════════════════════════════════════════════════════════"
echo "  评测模型准备脚本"
echo "═══════════════════════════════════════════════════════════════"
echo "  训练 checkpoint:  $ckpt_dir"
echo "  BAGEL 原始路径:   $bagel_base_path"
echo "  目标临时目录:     $target_dir"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: 检查源目录是否存在 ──
echo ""
echo ">>> [Step 1/5] 检查源目录..."

if [ ! -d "$ckpt_dir" ]; then
    echo "  ❌ 错误: 训练 checkpoint 目录不存在: $ckpt_dir"
    exit 1
fi
echo "  ✅ 训练 checkpoint 目录存在"

if [ ! -d "$bagel_base_path" ]; then
    echo "  ❌ 错误: BAGEL 原始模型路径不存在: $bagel_base_path"
    exit 1
fi
echo "  ✅ BAGEL 原始模型路径存在"

# ── Step 2: 创建目标目录 ──
echo ""
echo ">>> [Step 2/5] 创建目标目录: $target_dir"
mkdir -p "$target_dir"
echo "  ✅ 目录已创建"

# ── Step 3: 复制模型权重文件 ──
echo ""
echo ">>> [Step 3/5] 复制模型权重文件..."

weights_copied=0
for weight_file in ema.safetensors model.safetensors; do
    src="$ckpt_dir/$weight_file"
    dst="$target_dir/$weight_file"
    if [ -f "$src" ]; then
        if [ -f "$dst" ]; then
            src_size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null)
            dst_size=$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst" 2>/dev/null)
            # if [ "$src_size" = "$dst_size" ]; then
            #     echo "  ⏭️  跳过 $weight_file (已存在且大小一致: ${src_size} bytes)"
            #     weights_copied=$((weights_copied + 1))
            #     continue
            # fi
        fi
        echo "  📦 复制 $weight_file ($(du -h "$src" | cut -f1))..."
        cp "$src" "$dst"
        echo "  ✅ $weight_file 复制完成"
        weights_copied=$((weights_copied + 1))
    else
        echo "  ⚠️  $weight_file 不存在于 $ckpt_dir，跳过"
    fi
done

if [ "$weights_copied" -eq 0 ]; then
    echo "  ❌ 错误: 没有找到任何模型权重文件 (ema.safetensors 或 model.safetensors)"
    exit 1
fi
echo "  ✅ 共复制 $weights_copied 个权重文件"

# ── Step 4: 复制配置文件 ──
echo ""
echo ">>> [Step 4/5] 从 BAGEL 原始路径复制配置文件..."

# 需要复制的配置文件列表
config_files=(
    config.json
    llm_config.json
    vit_config.json
    generation_config.json
    preprocessor_config.json
    tokenizer_config.json
    tokenizer.json
    vocab.json
    merges.txt
    ae.safetensors
)

copied_count=0
skipped_count=0
missing_count=0

for f in "${config_files[@]}"; do
    src="$bagel_base_path/$f"
    dst="$target_dir/$f"
    if [ -f "$dst" ]; then
        echo "  ⏭️  跳过 $f (目标已存在)"
        skipped_count=$((skipped_count + 1))
    elif [ -f "$src" ]; then
        echo "  📦 复制 $f ($(du -h "$src" | cut -f1))..."
        cp "$src" "$dst"
        copied_count=$((copied_count + 1))
    else
        echo "  ⚠️  $f 在 BAGEL 原始路径中不存在，跳过"
        missing_count=$((missing_count + 1))
    fi
done

echo "  ✅ 新复制 $copied_count 个, 已存在跳过 $skipped_count 个, 缺失 $missing_count 个"

# ── Step 5: 验证文件完整性 ──
echo ""
echo ">>> [Step 5/5] 验证文件完整性..."

# 必需文件列表
required_files=(
    llm_config.json
    vit_config.json
    tokenizer_config.json
    tokenizer.json
    ae.safetensors
)

all_ok=true
for f in "${required_files[@]}"; do
    if [ -f "$target_dir/$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ 缺失必需文件: $f"
        all_ok=false
    fi
done

# 检查至少有一个权重文件
if [ -f "$target_dir/ema.safetensors" ]; then
    echo "  ✅ ema.safetensors (推荐用于评测)"
elif [ -f "$target_dir/model.safetensors" ]; then
    echo "  ✅ model.safetensors (可用于评测)"
else
    echo "  ❌ 缺失模型权重文件"
    all_ok=false
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
if $all_ok; then
    echo "  ✅ 所有文件准备完毕！"
else
    echo "  ⚠️  部分文件缺失，请检查上述输出"
fi
echo ""
echo "  目标目录: $target_dir"
echo "  文件列表:"
ls -lh "$target_dir/"
echo ""
echo "  在评测脚本中使用:"
echo "    model_path=\"$target_dir\" bash scripts/eval/run_behind_vae_re10k.sh"
echo "═══════════════════════════════════════════════════════════════"
