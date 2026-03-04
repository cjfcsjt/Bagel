import os
import yaml

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.g2vlm import (
    G2VLMConfig, 
    G2VLM, 
    Qwen2VLConfig,
    Qwen2VLForCausalLM,
    Dinov2WithRegistersConfig, Dinov2WithRegistersModel
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

from data.transforms import ImageTransform, InternVLImageTransform, QwenVL2ImageTransform
from data.transforms_vggt import DinoImageTransform, DinoImageNormalizeTransform

from modeling.qwen2vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from modeling.qwen2vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from modeling.g2vlm.qwen2vl import Qwen2VLForCausalLM
from modeling.qwen2vl.configuration_qwen2_vl import Qwen2VLVisionConfig

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import torchvision
import open3d as o3d  # Optional but recommended for .ply saving (install with `pip install open3d`)
import torch.nn.functional as F


def load_model_and_tokenizer_from_path(model_path):
    llm_config = Qwen2VLConfig.from_json_file(os.path.join(model_path, "text_config.json"))

    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = 'Qwen2VLMoTDecoderLayer'  

    vit_config = Qwen2VLVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.patch_size =14

    dino_config = Dinov2WithRegistersConfig.from_json_file(os.path.join(model_path, "dino_config.json"))

    config = G2VLMConfig(
        visual_und=True,
        visual_recon=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        dino_config=dino_config,
        vit_max_num_patch_per_side=36,
    )
    language_model = Qwen2VLForCausalLM(llm_config)
    vit_model      = Qwen2VisionTransformerPretrainedModel(vit_config)
    dino_model = Dinov2WithRegistersModel(dino_config)

    model = G2VLM(language_model, vit_model, dino_model, config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vit_image_transform = QwenVL2ImageTransform(768, 768, 14)
    dino_transform = DinoImageNormalizeTransform(target_size=518)

    model_state_dict_path = os.path.join(model_path, "model.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del model_state_dict
    model = model.cuda().eval()

    return model, tokenizer, new_token_ids , vit_image_transform, dino_transform

def load_model_and_tokenizer(args):
    llm_config = Qwen2VLConfig.from_json_file(os.path.join(args.model_path, "llm_config.json"))

    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = 'Qwen2VLMoTDecoderLayer'  

    # vit_config = Qwen2VLVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config = Qwen2VLVisionConfig.from_pretrained(args.vit_path)
    vit_config.patch_size =14

    dino_config = Dinov2WithRegistersConfig.from_json_file(os.path.join(args.model_path, "dino_config.json"))

    config = G2VLMConfig(
        visual_und=True,
        visual_recon=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        dino_config=dino_config,
        vit_max_num_patch_per_side=36,
    )
    language_model = Qwen2VLForCausalLM(llm_config)
    vit_model      = Qwen2VisionTransformerPretrainedModel(vit_config)
    dino_model = Dinov2WithRegistersModel(dino_config)

    model = G2VLM(language_model, vit_model, dino_model, config)

    # tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.vit_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vit_image_transform = QwenVL2ImageTransform(768, 768, 14)
    dino_transform = DinoImageNormalizeTransform(target_size=518)

    model_state_dict_path = os.path.join(args.model_path, "model.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del model_state_dict
    model = model.cuda().eval()

    return model, tokenizer, new_token_ids , vit_image_transform, dino_transform

def build_transform(pixel=224):
    image_transform = QwenVL2ImageTransform(pixel, pixel, 14)

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation



def save_ply_visualization(
    pred_dict, save_path, init_conf_threshold=20.0, filter_nan=True, verbose=True):
    """
    Saves a PLY visualization of the world points and images.
    Args:
    """
    pred_dict = pred_dict.copy()
    for key in pred_dict.keys():
        if isinstance(pred_dict[key], torch.Tensor):
            pred_dict[key] = pred_dict[key].detach()
    
    index = 0
    seq_len = len(pred_dict["images"][index])  

    images = pred_dict["images"][index]
    images_grid = torchvision.utils.make_grid(
        images, nrow=8, normalize=False, scale_each=False
    )
    images_grid = images_grid
    images_save_path = f'input_images.png'
    torchvision.utils.save_image(
        images_grid, images_save_path, normalize=False, scale_each=False
    )
    
    pred_pts = pred_dict["points"][index]  
    
    data_h, data_w = images.shape[-2:]
    global_points = pred_pts

    data_size = (data_h, data_w)
    global_points = F.interpolate(
        global_points.permute(0, 3, 1, 2), data_size,
        mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1)  

    global_points = global_points.cpu().numpy()
    pred_pts = global_points
    
    colors = images.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, 3)

    pred_pts = pred_pts.reshape(-1, 3)

    
    if filter_nan:
        nan_mask = np.any(np.isnan(pred_pts), axis=1)
        inf_mask = np.any(np.isinf(pred_pts), axis=1)
        invalid_mask = np.logical_or(nan_mask, inf_mask)
        
        valid_mask = ~invalid_mask
        
        total_points = len(pred_pts)
        invalid_count = np.sum(invalid_mask)
        valid_count = np.sum(valid_mask)
        
        if invalid_count > 0:
            pred_pts_filtered = pred_pts[valid_mask]
            colors_filtered = colors[valid_mask]
            
            pred_pts = pred_pts_filtered
            colors = colors_filtered
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)
 