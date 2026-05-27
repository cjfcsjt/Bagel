# 改动文档：部分加噪重建模式（partial noise）与 behind_vae 评估流程

对应提交：`新增部分加噪重建模式与 behind_vae 评估流程`

本次改动围绕两条主线展开：

1. **训练侧**：新增 `use_partial_noise` 部分加噪重建训练模式，配套 `no_ce_loss` 纯重建开关，并重写 `edit_recon` / `edit_recon_mae` 数据集以支持 **behind_vae** 序列布局。
2. **评估侧**：新增一整套 behind_vae / MAE / prefix 的图像生成评估脚本，以及 webdataset / parquet 数据转换工具。

涉及 27 个文件，约 +10591 / -299 行。下面按功能模块说明。

---

## 1. 训练逻辑与配置

### 1.1 `train/pretrain_unified_navit_vae.py`
- 新增 `TrainingArguments.no_ce_loss`（默认 `False`）：开启后关闭所有文本 cross-entropy 损失，进入**纯重建模式**。
- 把 `behind_vae`、`no_ce_loss` 两个开关注入到 `dataset_config`，传递给下游数据集。
- 修复 wandb 配置上报：`wandb.config.update(...)` 增加 `allow_val_change=True`，避免恢复训练时重复写入 config 报错。

### 1.2 `scripts/train_sft_vae.sh`（训练启动脚本切换）
- GPU 由 8 卡切到 4 卡：`CUDA_VISIBLE_DEVICES=4,5,6,7`、`--nproc_per_node=4`、`--num_shard 4`。
- Token 预算上调：`max_num_tokens 52000→64000`、`expected_num_tokens 50000→63000`、`max_num_tokens_per_sample 45000→60000`。
- 训练模式开关切换：
  - 关闭旧 masking：`--use_masking False`、`--use_mae_masking False`
  - 开启新模式：`--use_partial_noise True`、`--no_ce_loss True`、`--freeze_und False`
  - `--behind_vae False`（本次脚本默认走 two-pass，behind_vae 由数据集分支控制）
- 模型缓存路径由 `/tmp/.cache` 改为 `/root/.cache`。
- 新 `RUNID=171`，checkpoint 目录改为 `joint_vae_geo_videollm3d_partial_masking_denoise_vae_prefix_${RUNID}`。

### 1.3 `data/configs/joint_train.yaml`
- 数据集段 `mae_recon` 重命名为 `recon`。
- `videollm3d` 数据路径由 `/mnt/group/...` 迁移到 `/apdcephfs_303747097/share_303747097/...`（`video_folder` / `annotation_dir` / `metadata_dir`）。

---

## 2. 数据集：edit_recon 与 behind_vae 序列构造

### 2.1 `data/interleave_datasets/edit_recon_dataset.py`
`MaskedReconIterableDataset` 新增 `behind_vae` 参数，`_parse_row` 按模式分支：

- **two-pass 模式（`_parse_row_two_pass`，原始逻辑保留）**
  ```
  Pass 1（无 loss，clean condition）:
    [ref_i_vae_clean][ref_i_vit] ...        # ref：clean VAE + VIT
    [nonref_j_vae_clean] ...                 # nonref：仅 clean VAE（不加 VIT，避免泄漏被 mask 区域）
  Pass 2（有 loss，加噪重建）:
    [nonref_j_vae_noise] ...                 # nonref：噪声 VAE token，attend 前面所有 clean 条件做重建
  ```

- **behind_vae 模式（`_parse_row_behind_vae`，新增）**
  ```
  [text: system + user prompt]
    [VIT: ref_1 ... ref_K]                   # 所有 ref 帧 VIT，vit_type='ref'，完全可见
    [VIT: nonref_1 ... nonref_M]             # 所有 nonref 帧 VIT，vit_type='nonref'，被 mask 的 VIT 对 VAE 不可见
    [VAE: nonref_1 ... nonref_M (masked denoise)]   # 仅 nonref 帧有 VAE，放在 VIT 之后，loss=1
  [text: assistant]
  ```
  - 使用 `apply_template_qwenvl2` 基于模板构造 `<vit_image>` / `<vae_image>` 占位符序列（所有帧 VIT + 仅 nonref 帧 VAE）。
  - 用 `vit_counter` / `vae_counter` 遍历 `split_list`，分别填充 text / vit_image / vae_image 的 `sequence_plan`。

### 2.2 `data/interleave_datasets/edit_recon_mae_dataset.py`
- 同步引入 `behind_vae` 模式与基于模板的 VAE 序列构造（与 edit_recon 对应的 MAE 变体）。

### 2.3 `data/dataset_base_vae.py`、`data/dataset_info.py`、`data/videollm3d_dataset.py`
- 配合 `behind_vae` / `no_ce_loss` 透传与数据集注册、路径调整。
- **修复 VAE→VIT 的 mask 跨分辨率映射**：改用 VIT 图片的实际像素尺寸来做 mask 映射，避免 VAE 与 VIT 分辨率不一致时映射错位。

---

## 3. 模型推理：带 mask 的 VIT KV cache 更新

### 3.1 `modeling/bagel/bagel.py`
- 新增 `forward_cache_update_vit_masked(...)`：推理时**带 mask 地更新 VIT 的 KV cache**。
  - VIT encoder 仍处理**完整图片**（保证 VIT 内部 self-attention 上下文完整）。
  - 在 VIT 输出写入 LLM KV cache **之前**，把被 mask 的 VIT patch 从 packed_sequence 中剔除（不写入 KV cache）。
  - 效果等价于训练时用 attention mask 屏蔽这些 token：被 mask token 的 KV 不在 cache 中，VAE token 自然无法 attend 到它们。
  - 重建 `packed_sequence` 布局为 `[start_of_image][visible_vit...][end_of_image]`，并重算 `position_ids` / `indexes`；全部被 mask 时退化为仅注入 start/end_of_image 文本 token；`vit_mask_flat=None` 时回退到原始 `forward_cache_update_vit`。返回 `(past_key_values, num_visible)`。
- 鲁棒性修复：3 处 `self.language_model.model.enable_taylorseer` 改为 `getattr(self.language_model.model, 'enable_taylorseer', False)`，避免模型无该属性时报错。

### 3.2 `modeling/bagel/bagel_multipass.py`（新增，约 +1457 行）
- 新增多遍（multipass）版本的 Bagel 推理/前向实现，配合 behind_vae 与部分加噪重建的多阶段序列处理。

### 3.3 `modeling/g2vlm/dinov2_model.py`
- DINOv2 改用新的 autocast API。

---

## 4. 评估流程（eval）

### 4.1 图像生成脚本 `eval/gen/`（新增）
- `gen_images_behind_vae_re10k.py`：RE10K 数据集上的 behind_vae 生成评估。
- `gen_images_behind_vae_videollm3d.py`：VideoLLM3D 上的 behind_vae 生成评估。
- `gen_images_mae_re10k.py`：RE10K 上的 MAE 重建生成评估。
- `gen_images_prefix_vae_videollm3d.py`：VideoLLM3D 上的 prefix VAE 生成评估。

### 4.2 评估启动脚本 `scripts/eval/`（新增）
- `prepare_eval_model.sh`：评估前的模型准备。
- `run_behind_vae_re10k.sh`、`run_behind_vae_videollm3d.sh`、`run_mae_re10k.sh`、`eval_behind_vae_re10k.sh`：各评估流程的启动入口。

### 4.3 数据转换工具 `eval/spatial_reason/unify/`（新增 / 大幅扩展）
- `unify_to_parquet.py`：大幅扩展（约 +2996 行）的 parquet 统一转换。
- `to_webdataset_re10k.py`、`to_webdataset_blendedmvs.py`、`to_webdataset_scannet_new.py`、`to_webdataset_scannetpp_load.py`：各数据集到 webdataset 的转换脚本。
- `to_wds_re10k.sh`、`to_wds_blendedmvs.sh`：对应转换启动脚本。

---

## 5. 模式速查

| 模式 | 开关 | VIT 条件 | VAE token | 文本 loss | 序列布局 |
|------|------|----------|-----------|-----------|----------|
| two-pass 部分加噪 | `use_partial_noise=True`, `behind_vae=False` | 仅 ref 帧 | nonref 噪声 VAE（两遍） | 由 `no_ce_loss` 控制 | clean condition → noise VAE |
| behind_vae | `behind_vae=True` | 所有帧（ref + nonref） | 仅 nonref，置于 VIT 之后 | 由 `no_ce_loss` 控制 | text → VIT(all) → VAE(nonref) → text |

> `no_ce_loss=True` 时进入纯重建模式，只保留 VAE 重建（MSE）损失，关闭全部文本 CE 损失。
