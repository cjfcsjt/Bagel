# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# 基于 Ray Train (TorchTrainer) 的分布式训练脚本
# 用法: python train/ray_pretrain_unified_navit_vae.py --num_ray_workers 8 ...
# 设置环境变量，确保每个 worker 继承
import ray
import os
runtime_env = {
    "env_vars": {
        "PATH": "/mnt/group/jingfanchen/miniconda3/bin:/mnt/group/jingfanchen/miniconda3/condabin:" + os.environ.get("PATH", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "NCCL_ASYNC_ERROR_HANDLING": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", ""),
        "HF_HOME": os.environ.get("HF_HOME", ""),
    },
    "py_executable": "/mnt/group/jingfanchen/miniconda3/envs/ray_py311/bin/python",

}
# 初始化 Ray 集群连接
ray.init(address="auto", runtime_env=runtime_env)
    
import functools
import gc
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

import ray
import ray.train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchConfig, get_device

from data.dataset_base_vae import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils_vae import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """根据 GPU 型号猜测单卡 BF16 TFLOPs；未知型号则返回默认值。"""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )

    # --- masked reconstruction ---
    use_masking: bool = field(
        default=False,
        metadata={"help": "Enable masked reconstruction mode for conditional generation."}
    )
    use_mae_masking: bool = field(
        default=False,
        metadata={"help": "Enable masked autoencoder (MAE) style masking for conditional generation."}
    )
    mask_mode: str = field(
        default=None,
        metadata={"help": "List of mask generation strategies, e.g. ['random', 'rectangle', 'blob']. "
                  "One is randomly selected per forward pass. Default: ['random']."}
    )
    mask_ratio: str = field(
        default=None,
        metadata={"help": "List of mask ratios corresponding to each mask_mode entry, e.g. [0.75, 0.5]. "
                  "Default: [0.75]."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    # --- Ray 分布式配置 ---
    num_ray_workers: int = field(
        default=8,
        metadata={"help": "Ray Train 启动的总 worker 数（= 总 GPU 数）。"}
    )
    num_ray_gpus_per_worker: int = field(
        default=1,
        metadata={"help": "每个 Ray worker 使用的 GPU 数量。"}
    )
    num_ray_cpus_per_worker: int = field(
        default=8,
        metadata={"help": "每个 Ray worker 使用的 CPU 数量。"}
    )
    ray_address: str = field(
        default=None,
        metadata={"help": "Ray 集群地址，例如 'auto' 或 'ray://<head_ip>:10001'。为 None 则本地启动。"}
    )


# =====================================================================
# Ray Train worker 函数：每个 GPU worker 执行此函数
# =====================================================================
def train_func(config: dict):
    """
    由 Ray TorchTrainer 在每个 worker 上调用的训练函数。
    config 字典包含序列化后的 model_args, data_args, training_args。
    """
    import torch.distributed as dist

    # ---- 从 config 中恢复参数 ----
    model_args = config["model_args"]
    data_args = config["data_args"]
    training_args = config["training_args"]

    # ---- 获取 Ray 分布式上下文 ----
    # Ray Train 已经自动初始化了 torch.distributed 进程组
    global_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    device = get_device()
    torch.cuda.set_device(device)

    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # ---- 日志设置 ----
    if global_rank == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, global_rank)
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(vars(training_args))
        wandb.config.update(vars(model_args))
        wandb.config.update(vars(data_args))
        # 保存训练脚本和数据集配置文件到 wandb，方便复现
        wandb.save(os.path.abspath(__file__), policy="now")
        wandb.save(os.path.abspath(data_args.dataset_config_file), policy="now")
        wandb.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset_info.py"), policy="now")
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, global_rank)
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # ---- 自动恢复逻辑 ----
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # ---- 设置随机种子 ----
    seed = training_args.global_seed * world_size + global_rank
    set_seed(seed)

    # ---- 构建模型 ----
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    bagel_config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        use_masking=training_args.use_masking,
        use_mae_masking=training_args.use_mae_masking,
        mask_mode=training_args.mask_mode.split(',') if training_args.mask_mode else None,
        mask_ratio=[float(r) for r in training_args.mask_ratio.split(',')] if training_args.mask_ratio else None,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        bagel_config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # ---- Tokenizer ----
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # ---- 冻结模块 ----
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # ---- FSDP 封装 & 加载预训练权重 ----
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if global_rank == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # ---- 优化器 & 调度器 ----
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    # ---- 恢复优化器/调度器/训练步数 ----
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # ---- 数据集 & DataLoader ----
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=global_rank,
        world_size=world_size,
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # ---- 准备训练 ----
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    # ---- 训练循环 ----
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    curr_step = train_step  # 确保在循环外也有定义

    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                loss_dict = fsdp_model(**data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * world_size / total_ce_loss_weights
            else:
                ce = ce.sum() * world_size / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            if training_args.use_masking or training_args.use_mae_masking:
                total_mse_tokens = torch.tensor(mse.shape[0], device=device)
            else:
                total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * world_size / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
        
        # ---- 日志记录 ----
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * world_size
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if global_rank == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if global_rank == 0:
                wandb.log(wandb_log, step=curr_step)

            # 通过 ray.train.report 上报指标（可选，用于 Ray Dashboard 监控）
            ray.train.report(metrics={
                "step": curr_step,
                "ce_loss": wandb_log.get("ce", 0.0),
                "mse_loss": wandb_log.get("mse", 0.0),
                "mfu": mfu_value,
                "tokens_per_sec": tokens_per_sec,
            })

            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        # ---- 更新 data_status ----
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        # ---- 保存 checkpoint ----
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if global_rank == 0:
                gather_list = [None] * world_size
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if global_rank != 0 else [data_status] * world_size

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---- 保存最终 checkpoint ----
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if global_rank == 0:
            gather_list = [None] * world_size
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if global_rank != 0 else [data_status] * world_size
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if global_rank == 0:
        wandb.finish()


# =====================================================================
# 主入口：解析参数 → 初始化 Ray → 启动 TorchTrainer
# =====================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"[Ray] 集群资源: {ray.cluster_resources()}")
    print(f"[Ray] 启动 {training_args.num_ray_workers} 个 worker，"
          f"每个 worker {training_args.num_ray_gpus_per_worker} GPU, "
          f"{training_args.num_ray_cpus_per_worker} CPU")

    # 构建 Ray Train 的 ScalingConfig
    scaling_config = ScalingConfig(
        num_workers=training_args.num_ray_workers,
        use_gpu=True,
        resources_per_worker={
            "GPU": training_args.num_ray_gpus_per_worker,
            "CPU": training_args.num_ray_cpus_per_worker,
        },
    )

    # 构建 RunConfig（Ray Train 的运行配置）
    # run_config = RunConfig(
    #     name=f"bagel-ray-{training_args.wandb_runid}",
    #     storage_path=training_args.results_dir,
    #     # checkpoint_config=CheckpointConfig(
    #     #     num_to_keep=3,  # 保留最近 3 个 Ray checkpoint
    #     # ),
    # )

    # 需要传递给每个 worker 的配置（序列化为 dict）
    train_loop_config = {
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args,
    }

    # 创建 TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        # run_config=run_config,
        torch_config=ray.train.torch.TorchConfig(
            backend="nccl",
        ),
    )

    # 启动训练
    print("[Ray] 开始训练...")
    result = trainer.fit()
    print(f"[Ray] 训练完成! 结果: {result}")

    ray.shutdown()


if __name__ == "__main__":
    main()