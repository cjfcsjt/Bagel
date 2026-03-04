import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.g2vlm.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding, PositionEmbedding_Extra

from modeling.g2vlm.qwen2vl import (
    Qwen2VLDecoderLayer, 
    Qwen2VLMoEDecoderLayer, 
    Qwen2VLMoTDecoderLayer,
)
from modeling.g2vlm.dinov2_model import Dinov2WithRegistersLayer, Dinov2WithRegistersEncoder, Dinov2WithRegistersEmbeddings, Dinov2WithRegistersModel
from modeling.dinov3.dinov3_model import DINOv3ViTLayer, DINOv3ViTEmbeddings,DINOv3ViTRopePositionEmbedding, DINOv3ViTModel
from modeling.qwen2vl.modeling_qwen2_vl import PatchEmbed, VisionRotaryEmbedding, Qwen2VLVisionBlock, PatchMerger, Qwen2VisionTransformerPretrainedModel

import shutil
import socket
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from collections import defaultdict
import gc
import pprint
import pandas as pd
import time
from datetime import datetime 
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import FileSystemReader

def save_latest_checkpoints(ckpt_dir, keep_latest=2):
    """
    Keeps only the latest 'keep_latest' checkpoints in ckpt_dir.
    Assumes checkpoint folders are named with step numbers like 0001000/.
    """
    if dist.get_rank() != 0:
        return

    # List all subdirectories (assumed to be step-based checkpoint dirs)
    steps = []
    for d in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit():
            steps.append(int(d))
    
    steps.sort()

    while len(steps) > keep_latest:
        oldest_step = steps.pop(0)
        oldest_ckpt_path = os.path.join(ckpt_dir, f"{oldest_step:07d}")
        shutil.rmtree(oldest_ckpt_path)
        print(f"Deleted old checkpoint: {oldest_ckpt_path}")
        

class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):

    device_id = dist.get_rank() % torch.cuda.device_count()
    target_device = torch.device(f"cuda:{device_id}")  # Use same device as FSDP
    # Explicitly move ignored modules to the target device
    for module in ignored_modules:
        module.to(target_device)  

    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
        print("device_mesh shape:", device_mesh.mesh.shape)
    else:
        device_mesh = None
 
    
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2VLDecoderLayer,
                Qwen2VLMoEDecoderLayer,
                Qwen2VLMoTDecoderLayer,
                Dinov2WithRegistersLayer,
                Dinov2WithRegistersModel,
                Dinov2WithRegistersEmbeddings,
                DINOv3ViTLayer, DINOv3ViTEmbeddings,DINOv3ViTRopePositionEmbedding, DINOv3ViTModel,
                MLPconnector,
                PatchEmbed, VisionRotaryEmbedding, Qwen2VLVisionBlock, PatchMerger,
                Qwen2VisionTransformerPretrainedModel,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16, 
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device_id,  
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )
    

class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_fsdp_ckpt(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scaler,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            try:
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    ema_state_dict = ema_model.state_dict() 
                    if dist.get_rank() == 0:
                        save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))
                    del ema_state_dict
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)
        ):
            model_state_dict = model.state_dict()
            model_writer = FileSystemWriter(os.path.join(save_path, "model"))
            dcp.save(state_dict=model_state_dict, storage_writer=model_writer)
            del model_state_dict
            gc.collect()
            torch.cuda.empty_cache()

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_path, "scaler.pt"))
            
        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))


        if data_status is not None:
            if fsdp_config.sharding_strategy == "HYBRID_SHARD":
                os.makedirs(os.path.join(save_path, "data_status"), exist_ok=True)
                torch.save(
                    data_status, os.path.join(save_path, "data_status", f"rank{dist.get_rank()}.pt")
                )
                del data_status
                gc.collect()
                torch.cuda.empty_cache()
            elif fsdp_config.sharding_strategy == "FULL_SHARD":
                if dist.get_rank() == 0 and data_status is not None:
                    torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_fsdp_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_file = os.path.join(resume_from, "ema.safetensors")
                shard_dir = os.path.join(resume_from, "ema")
            else:
                model_file = os.path.join(resume_from, "model.safetensors")
                shard_dir = os.path.join(resume_from, "model")

            assert isinstance(model, FSDP)
            if os.path.exists(model_file):
                # 单文件 safetensors
                logger.info(f"Detected safetensors checkpoint for main model: {model_file}")
                state_dict = load_file(model_file, device="cpu")
                for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                    if key in state_dict:
                        del state_dict[key]
                msg = model.load_state_dict(state_dict, strict=False)
                logger.info(msg)
                del state_dict
                gc.collect()
                torch.cuda.empty_cache()
            elif os.path.exists(shard_dir):
                logger.info(f"Detected sharded checkpoint for main model: {shard_dir}")
                model_reader = FileSystemReader(shard_dir)
                with FSDP.state_dict_type(
                    model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),
                ):
                    model_state_dict = model.state_dict()
                    dcp.load(state_dict=model_state_dict, storage_reader=model_reader)
                    for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                        if key in model_state_dict:
                            model_state_dict.pop(key)
                    msg = model.load_state_dict(model_state_dict, strict=False)
                    logger.info(msg)
                    del model_state_dict
                    gc.collect()
                    torch.cuda.empty_cache()

            if ema_model is not None:
                ema_file = os.path.join(resume_from, "ema.safetensors")
                ema_shard_dir = os.path.join(resume_from, "ema")
                assert isinstance(ema_model, FSDP)
                if os.path.exists(ema_file):
                    logger.info(f"Detected safetensors checkpoint for EMA model: {ema_file}")
                    ema_state_dict = load_file(ema_file, device="cpu")
                    for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                        if key in ema_state_dict:
                            ema_state_dict.pop(key)
                    msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                    logger.info(msg)
                    del ema_state_dict
                    gc.collect()
                    torch.cuda.empty_cache()
                elif os.path.exists(ema_shard_dir):
                    files = [f for f in os.listdir(ema_shard_dir) if f.endswith(".pt") or f.endswith(".safetensors")]
                    if len(files) > 1:
                        logger.info(f"Detected sharded checkpoint for EMA model: {ema_shard_dir}")
                        reader = FileSystemReader(ema_shard_dir)
                        with FSDP.state_dict_type(
                            ema_model,
                            StateDictType.SHARDED_STATE_DICT,
                            ShardedStateDictConfig(offload_to_cpu=True)
                        ):
                            ema_state_dict = ema_model.state_dict()
                            dcp.load(ema_state_dict, reader)
                            for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                                ema_state_dict.pop(key, None)
                            msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                            logger.info(msg)
                        del ema_state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    elif len(files) == 1:
                        ckpt_file = os.path.join(ema_shard_dir, files[0])
                        logger.info(f"Detected rank0-only FULL_STATE_DICT checkpoint for EMA model: {ckpt_file}")
                        with FSDP.state_dict_type(
                            ema_model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        ):
                            if dist.get_rank() == 0:
                                ema_state_dict = load_file(ckpt_file, device="cpu")
                                for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                                    ema_state_dict.pop(key, None)
                            else:
                                ema_state_dict = None
                            ema_state_dict = FSDP.broadcast_state_dict(ema_model, ema_state_dict, src=0)
                            msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                            logger.info(msg)
                        del ema_state_dict
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        logger.info("EMA directory exists but no valid checkpoint files found.")
                else:
                    logger.info("No EMA checkpoint found; initializing EMA model from main model.")
                
        else:
            logger.info("Training from scratch.")
        return model, ema_model

    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scaler,
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError
            
        ########scaler is global, not sharded like optimizer 
        if dist.get_rank() == 0 and scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_path, "scaler.pt"))

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt_except_moe(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            
            filtered_model_state_dict = {k: v for k, v in model_state_dict.items() 
                                        if 'moe' not in k}
            
            filtered_keys = [k for k in model_state_dict if 'moe' in k]
            if filtered_keys:
                logger.info(f"Filtered {len(filtered_keys)} MOE layers from checkpoint: {filtered_keys[:5]} (showing first 5)")
            

            msg = model.load_state_dict(filtered_model_state_dict, strict=False)
            logger.info(f"Loaded model weights: {msg}")
            del model_state_dict, filtered_model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"Replicating EMA model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                
                filtered_ema_state_dict = {k: v for k, v in ema_state_dict.items() 
                                        if 'moe' not in k}

                msg = ema_model.load_state_dict(filtered_ema_state_dict, strict=False)
                logger.info(f"Loaded EMA model weights: {msg}")
                del ema_state_dict, filtered_ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model


    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            # model_state_dict.pop('latent_pos_embed.pos_embed')
            # model_state_dict.pop('vit_pos_embed.pos_embed')
            # if 'dino_pos_embed.pos_embed' in model_state_dict:
            #     model_state_dict.pop('dino_pos_embed.pos_embed')
            
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                # ema_state_dict.pop('latent_pos_embed.pos_embed')
                # ema_state_dict.pop('vit_pos_embed.pos_embed')
                # if 'dino_pos_embed.pos_embed' in ema_state_dict:
                #     ema_state_dict.pop('dino_pos_embed.pos_embed')

                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scaler, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            # scaler_state_dict_path = os.path.join(resume_from, "scaler.pt")
            # scaler_state_dict = torch.load(scaler_state_dict_path, weights_only=True, map_location="cpu")
            # scaler.load_state_dict(scaler_state_dict)
            # del scaler_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scaler, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2VLDecoderLayer, 
        Dinov2WithRegistersLayer,
        DINOv3ViTLayer, #DINOv3ViTEmbeddings,DINOv3ViTRopePositionEmbedding, DINOv3ViTModel
        MLPconnector, 
        PatchMerger,
        Qwen2VLVisionBlock, 
        Qwen2VLMoEDecoderLayer, 
        Qwen2VLMoTDecoderLayer
    )
    if isinstance(module, module_options):
        print(f"[Checkpoint] Will checkpoint module: {module.__class__.__name__}")
        return True
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)
