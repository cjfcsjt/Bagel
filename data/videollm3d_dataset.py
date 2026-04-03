"""
VideoLLM3D Dataset – 基于 Bagel vlm_dataset 架构，集成 VEGA-3D VideoProcessor 和 task-grouped shuffle 策略。

核心特性：
  1. 继承 DistributedIterableDataset（与 SftJSONLIterableDataset 同级）
  2. 从 video_utils 导入 VEGA-3D 的 VideoProcessor：
     - 帧采样（uniform / mc 策略）
     - 世界坐标计算（在线 unproject：深度图 + 位姿 → 世界坐标）
     - 生成式特征加载（离线加载 Seva/VJEPA/Wan 等预提取特征）
     - 物体框加载（GT/pred box）
  3. 在 get_data_paths() 中实现 VEGA-3D 的 task-grouped shuffle：
     - 按任务类型分组：QA(scanqa+sqa3d)、Caption(scan2cap)、Grounding(scanrefer+multi3drefer)
     - 每组内部按 text length 分组 shuffle
     - 切分 megabatch → 跨组 megabatch 随机 shuffle
  4. while True 循环开头 re-shuffle（每轮 repeat 重新打乱）
  5. __iter__ 中处理 VEGA-3D llava_style JSON 格式，输出 Bagel sequence_plan 格式
"""

import json
import os
import random
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, PngImagePlugin

from .video_utils import VideoProcessor, merge_video_dict
from .data_utils import pil_img2rgb, apply_template_qwenvl2
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


# ═══════════════════════════════════════════════════════════════════════
#  VEGA-3D 任务分组映射
# ═══════════════════════════════════════════════════════════════════════

# 与 VEGA-3D train_3d.py 中的 task_mapping 一致
TASK_MAPPING = {
    "scanqa":       0,   # QA 任务
    "sqa3d":        0,   # QA 任务
    "scan2cap":     1,   # Caption 任务
    "scanrefer":    2,   # Grounding 任务
    "multi3drefer": 2,   # Grounding 任务
}


def _infer_task_id_from_filepath(filepath: str) -> Optional[int]:
    """根据 JSONL 文件名推断 task_id。"""
    basename = os.path.basename(filepath).lower()
    for key, tid in TASK_MAPPING.items():
        if key in basename:
            return tid
    return None


def _infer_task_id_from_item(item: dict) -> int:
    """根据单条数据的 metadata.dataset 字段推断 task_id。"""
    meta = item.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    ds_name = meta.get("dataset", "").lower()
    return TASK_MAPPING.get(ds_name, 0)


def _compute_text_length(item: dict) -> int:
    """计算单条数据的 text token 长度（按空格分词，与 VEGA-3D 一致）。"""
    total = 0
    for conv in item.get("conversations", []):
        total += len(conv.get("value", "").split())
    return max(total, 1)


# ═══════════════════════════════════════════════════════════════════════
#  VEGA-3D task-grouped shuffle 核心算法
#  移植自 llava_trainer.py 的 get_task_length_grouped_indices
# ═══════════════════════════════════════════════════════════════════════

def _split_to_even_chunks(indices, lengths, num_chunks):
    """将索引列表按长度均匀分配到 num_chunks 个桶中。"""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]

    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def _get_length_grouped_indices(lengths, batch_size, world_size, rng_obj):
    """
    按长度分组的 shuffle（与 VEGA-3D 的 get_length_grouped_indices 一致）。
    """
    indices = list(range(len(lengths)))
    rng_obj.shuffle(indices)

    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i:i + megabatch_size]
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(mb, key=lambda i: lengths[i], reverse=True)
        for mb in megabatches
    ]
    megabatches = [
        _split_to_even_chunks(mb, lengths, world_size)
        for mb in megabatches
    ]

    return [i for mb in megabatches for batch in mb for i in batch]


def task_grouped_shuffle(
    data_items: List[Tuple],
    task_ids: List[int],
    text_lengths: List[int],
    batch_size: int = 4,
    world_size: int = 1,
    rng_obj: Optional[random.Random] = None,
) -> List[Tuple]:
    """
    VEGA-3D task-grouped shuffle 策略。
    """
    if rng_obj is None:
        rng_obj = random.Random(42)

    task_indices: Dict[int, List[int]] = defaultdict(list)
    task_lengths: Dict[int, List[int]] = defaultdict(list)

    for i, (tid, tlen) in enumerate(zip(task_ids, text_lengths)):
        task_indices[tid].append(i)
        task_lengths[tid].append(tlen)

    task_shuffle: Dict[int, List[int]] = {}
    for tid in sorted(task_indices.keys()):
        group_indices = task_indices[tid]
        group_lengths = task_lengths[tid]
        local_order = _get_length_grouped_indices(
            group_lengths, batch_size, world_size, rng_obj
        )
        task_shuffle[tid] = [group_indices[j] for j in local_order]

    megabatch_size = world_size * batch_size
    task_megabatches: Dict[int, List[List[int]]] = {}
    for tid in sorted(task_shuffle.keys()):
        shuffled = task_shuffle[tid]
        task_megabatches[tid] = [
            shuffled[i:i + megabatch_size]
            for i in range(0, len(shuffled), megabatch_size)
        ]

    all_megabatches = []
    for tid in sorted(task_megabatches.keys()):
        mbs = task_megabatches[tid]
        if len(mbs) > 1:
            all_megabatches.extend(mbs[:-1])
        elif len(mbs) == 1:
            all_megabatches.append(mbs[0])

    rng_obj.shuffle(all_megabatches)

    ordered_indices = [i for mb in all_megabatches for i in mb]
    return [data_items[i] for i in ordered_indices]


def _load_json_or_jsonl(filepath: str) -> List[str]:
    """
    加载 JSON 或 JSONL 文件，统一返回 JSON 字符串列表。
    
    支持两种格式：
      - JSON 数组文件（VEGA-3D 的 llava_style.json）：整个文件是一个 JSON 数组
      - JSONL 文件：每行一个 JSON 对象
    """
    with open(filepath, 'r') as f:
        first_char = f.read(1).strip()
    
    if first_char == '[':
        # JSON 数组格式
        with open(filepath, 'r') as f:
            data_list = json.load(f)
        return [json.dumps(item, ensure_ascii=False) for item in data_list]
    else:
        # JSONL 格式（每行一个 JSON）
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]


# ═══════════════════════════════════════════════════════════════════════
#  VideoLLM3D Dataset
# ═══════════════════════════════════════════════════════════════════════

class VideoLLM3DIterableDataset(DistributedIterableDataset):
    """
    基于 Bagel vlm_dataset 架构的 Video3DLLM 数据集，集成 VEGA-3D VideoProcessor。

    与 SftJSONLIterableDataset 的区别：
      1. 使用 VEGA-3D VideoProcessor 处理 3D 视频数据（帧采样 + 世界坐标 + 生成式特征 + 物体框）
      2. 支持 VEGA-3D 的 task-grouped shuffle 策略
      3. while True 循环开头 re-shuffle（每轮 repeat 重新打乱）
      4. 数据格式为 VEGA-3D llava_style JSON（video 字段 + conversations）
      5. yield 的 dict 中包含 video_dict（world_coords, objects, boundary, generative_feature）

    Args:
        dataset_name:             数据集名称
        vit_transform:            VIT 图像变换
        tokenizer:                tokenizer
        frame_sampler:            帧采样器（仅用于非 3D 视频的 fallback）
        jsonl_path_list:          JSONL 文件路径列表
        data_dir_list:            图片目录列表
        num_used_data:            每个 JSONL 使用的数据量
        local_rank:               当前 rank
        world_size:               总 rank 数
        num_workers:              DataLoader worker 数
        data_status:              断点续训状态
        shuffle_seed:             随机种子
        task_grouped_shuffle:     是否启用 VEGA-3D task-grouped shuffle
        batch_size:               每个 rank 的 batch size
        video_folder:             视频数据根目录
        annotation_dir:           embodiedscan 标注目录
        metadata_dir:             元数据目录（box json 等）
        voxel_size:               体素大小
        min_xyz_range:            世界坐标最小范围
        max_xyz_range:            世界坐标最大范围
        frame_sampling_strategy:  帧采样策略（uniform / mc）
        val_box_type:             验证集 box 类型（gt / pred）
        force_sample:             是否强制采样指定帧数
        frames_upbound:           强制采样时的帧数上限
        generative_model_id:      生成式特征模型 ID
        generative_feature_source: 生成式特征来源（offline / none）
        add_spatial_instruction:  是否添加空间推理提示
    """

    def __init__(
        self,
        dataset_name,
        vit_transform,
        tokenizer,
        frame_sampler,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_seed=0,
        task_grouped_shuffle=True,
        batch_size=4,
        # ── VideoProcessor 扁平参数（直接从 YAML 传入） ──
        video_folder='data',
        annotation_dir='data/embodiedscan/',
        metadata_dir='data/metadata/',
        voxel_size=0.1,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy='uniform',
        val_box_type='pred',
        force_sample=False,
        frames_upbound=0,
        generative_model_id=None,
        generative_feature_source='none',
        add_spatial_instruction=False,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.shuffle_seed = shuffle_seed
        self.task_grouped_shuffle = task_grouped_shuffle
        self.batch_size = batch_size
        self._epoch_counter = 0

        # ── 初始化 VideoProcessor ──
        if min_xyz_range is None:
            min_xyz_range = [-15, -15, -5]
        if max_xyz_range is None:
            max_xyz_range = [15, 15, 5]
        self.video_processor = VideoProcessor(
            video_folder=video_folder,
            annotation_dir=annotation_dir,
            metadata_dir=metadata_dir,
            voxel_size=voxel_size,
            min_xyz_range=min_xyz_range,
            max_xyz_range=max_xyz_range,
            frame_sampling_strategy=frame_sampling_strategy,
            val_box_type=val_box_type,
        )
        self.force_sample = force_sample
        self.frames_upbound = frames_upbound
        self.generative_model_id = generative_model_id
        self.generative_feature_source = generative_feature_source
        self.add_spatial_instruction = add_spatial_instruction

        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_seed,
    ):
        """
        加载所有 JSON/JSONL 数据，并应用 VEGA-3D task-grouped shuffle 策略。
        支持 JSON 数组文件和 JSONL 文件两种格式。
        """
        data_paths = []
        task_ids = []
        text_lengths = []

        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            file_task_id = _infer_task_id_from_filepath(jsonl_path)

            raw_data = _load_json_or_jsonl(jsonl_path)
            raw_data = raw_data[:num_data_point]

            for json_str in raw_data:
                data_paths.append((json_str, image_dir))

                if self.task_grouped_shuffle:
                    try:
                        item = json.loads(json_str)
                        if file_task_id is not None:
                            tid = file_task_id
                        else:
                            tid = _infer_task_id_from_item(item)
                        tlen = _compute_text_length(item)
                    except Exception:
                        tid = 0
                        tlen = 1
                    task_ids.append(tid)
                    text_lengths.append(tlen)

        if self.task_grouped_shuffle and len(data_paths) > 0:
            rng_obj = random.Random(shuffle_seed)
            print(
                f"[VideoLLM3D] 应用 task-grouped shuffle: "
                f"{len(data_paths)} 条数据, "
                f"task 分布: { {tid: task_ids.count(tid) for tid in set(task_ids)} }"
            )
            data_paths = task_grouped_shuffle(
                data_paths, task_ids, text_lengths,
                batch_size=self.batch_size,
                world_size=self.world_size,
                rng_obj=rng_obj,
            )
        else:
            self.rng.seed(shuffle_seed)
            self.rng.shuffle(data_paths)

        return data_paths

    def _reshuffle_data_paths(self):
        """在 while True 循环的每轮 repeat 开头重新 shuffle。"""
        self._epoch_counter += 1
        new_seed = self.shuffle_seed + self._epoch_counter

        if self.task_grouped_shuffle:
            task_ids = []
            text_lengths = []
            for json_str, image_dir in self.data_paths_per_rank:
                try:
                    item = json.loads(json_str)
                    tid = _infer_task_id_from_item(item)
                    tlen = _compute_text_length(item)
                except Exception:
                    tid = 0
                    tlen = 1
                task_ids.append(tid)
                text_lengths.append(tlen)

            rng_obj = random.Random(new_seed)
            self.data_paths_per_rank = task_grouped_shuffle(
                self.data_paths_per_rank, task_ids, text_lengths,
                batch_size=self.batch_size,
                world_size=1,
                rng_obj=rng_obj,
            )
        else:
            rng_obj = random.Random(new_seed)
            rng_obj.shuffle(self.data_paths_per_rank)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            # ── re-shuffle：每轮 repeat 重新打乱（第一轮除外） ──
            if row_start_id == 0 and self._epoch_counter > 0:
                self._reshuffle_data_paths()
                data_paths_per_worker, _ = self.get_data_paths_per_worker()
                print(
                    f"[VideoLLM3D] rank-{self.local_rank} worker-{worker_id}: "
                    f"re-shuffled (epoch={self._epoch_counter})"
                )

            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(
                data_paths_per_worker_, start=row_start_id
            ):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []
                image_grid_thw_list = []

                try:
                    data_item = json.loads(data)
                    raw_images = None
                    video_dict = None

                    if 'image' in data_item:
                        # ── 单图或多图（与原逻辑一致） ──
                        if isinstance(data_item['image'], list):
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, img)))
                                for img in data_item['image']
                            ]
                        else:
                            raw_images = [
                                pil_img2rgb(Image.open(
                                    os.path.join(image_dir, data_item['image'])
                                ))
                            ]
                        special_tokens = '<vit_image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if item['from'] == "human":
                                if "<video>" in item['value']:
                                    item['value'] = item['value'].replace(
                                        "<video>", special_tokens
                                    ).strip()
                                else:
                                    item['value'] = special_tokens + " " + item['value']
                                break

                    elif 'video' in data_item:
                        # ── 3D 视频：使用 VideoProcessor 处理 ──
                        video_file = data_item['video']

                        # 处理 scan2cap 的 box_input
                        box_input = None
                        meta = data_item.get("metadata", {})
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except Exception:
                                meta = {}
                        dataset_name = meta.get("dataset", "").lower()

                        if dataset_name == "scan2cap":
                            box_input = data_item.get("box_input", [None, None, None])[:3]

                        # 添加空间推理提示（可选）
                        if self.add_spatial_instruction:
                            spatial_instruction = (
                                "The video captures 3D spatial information of a scene. "
                                "Please focus on the spatial relationships in the video "
                                "and answer the following questions."
                            )
                            for item in data_item['conversations']:
                                if item['from'] == 'human':
                                    if '<image>' in item['value']:
                                        item['value'] = item['value'].replace(
                                            '<image>',
                                            f'<image>\n{spatial_instruction}\n',
                                            1
                                        )
                                    else:
                                        item['value'] = f'{spatial_instruction}\n{item["value"]}'
                                    break

                        try:
                            # 使用 VideoProcessor 进行 3D 视频预处理
                            # 注意：不传 image_processor，使用默认 crop_size=384
                            video_dict = self.video_processor.preprocess(
                                video_file,
                                force_sample=self.force_sample,
                                frames_upbound=self.frames_upbound,
                                generative_model_id=self.generative_model_id,
                                generative_feature_source=self.generative_feature_source,
                            )
                            raw_images = video_dict.pop("images")
                            video_size = video_dict.pop("video_size")
                            video_dict["box_input"] = box_input

                        except Exception as e:
                            print(f"[VideoLLM3D] VideoProcessor 处理失败: {video_file}, 错误: {e}")
                            traceback.print_exc()
                            continue

                        # 替换 <image>/<video> 为 <vit_image> tokens
                        special_tokens = '<vit_image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if item['from'] == 'human':
                                if '<video>' in item['value']:
                                    item['value'] = item['value'].replace(
                                        '<video>', special_tokens
                                    )
                                elif '<image>' in item['value']:
                                    item['value'] = item['value'].replace(
                                        '<image>', special_tokens
                                    )
                                else:
                                    item['value'] = special_tokens + " " + item['value']
                                break

                except Exception:
                    traceback.print_exc()
                    continue

                # ── VIT 图像编码 ──
                if raw_images:
                    for raw_image in raw_images:
                        image_tensor, image_grid_thw = self.vit_transform(
                            [raw_image], img_num=len(raw_images)
                        )
                        image_tensor_list.append(image_tensor)
                        image_grid_thw_list.append(image_grid_thw[0])
                        num_tokens += image_tensor.shape[0] // 4

                # ── 文本模板处理 ──
                question = data_item['conversations'][0]["value"]
                answer = data_item['conversations'][1]["value"]
                split_list = apply_template_qwenvl2(
                    question_with_image_tokens=question, answer=answer
                )

                for item in split_list:
                    if item['type'] == 'text':
                        text_data = item['value']
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                'type': 'text',
                                'enable_cfg': 0,
                                'loss': item['loss'],
                                'special_token_loss': 0,
                                'special_token_label': None,
                            }
                            sequence_plan.append(current_plan)
                    elif item['type'] == 'vit':
                        current_plan = {
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                # ── 构建 yield dict ──
                result = dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    image_grid_thw_list=image_grid_thw_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

                # ── 附加 video_dict（3D 信息） ──
                if video_dict is not None:
                    result['video_dict'] = video_dict

                # ── 处理 scanrefer/multi3drefer 的 box_label ──
                if 'video' in data_item:
                    meta = data_item.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    dataset_name = meta.get("dataset", "").lower()
                    if dataset_name in ["scanrefer", "multi3drefer"]:
                        box_label = meta.get("object_id", [])
                        if not isinstance(box_label, list):
                            box_label = [box_label]
                        box_label = [int(i) for i in box_label]
                        result['box_label'] = box_label

                yield result

            # 一轮遍历完毕，重置 row_start_id，进入下一轮 repeat
            row_start_id = 0
            self._epoch_counter += 1
            print(
                f"{self.dataset_name} repeat in "
                f"rank-{self.local_rank} worker-{worker_id} "
                f"(epoch={self._epoch_counter})"
            )


# ═══════════════════════════════════════════════════════════════════════
#  Debug main 函数
# ═══════════════════════════════════════════════════════════════════════

def main():
    """
    Debug 入口：验证 VideoLLM3DIterableDataset 能否正常跑通。
    
    用法：
        cd /apdcephfs_303747097/share_303747097/jingfanchen/code/Bagel
        python -m data.videollm3d_dataset
    
    可通过环境变量控制：
        DATA_DIR:       VEGA-3D 数据根目录（默认 data）
        ANNOTATION_DIR: embodiedscan 标注目录（默认 data/embodiedscan/）
        JSONL_PATH:     单个 JSON/JSONL 文件路径（默认使用 scanqa）
        NUM_SAMPLES:    遍历的样本数（默认 5）
        STRATEGY:       帧采样策略（默认 uniform）
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Debug VideoLLM3DIterableDataset")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/"),
                        help="VEGA-3D 数据根目录")
    parser.add_argument("--annotation_dir", type=str, 
                        default=os.environ.get("ANNOTATION_DIR", "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/embodiedscan/"),
                        help="embodiedscan 标注目录")
    parser.add_argument("--jsonl_path", type=str,
                        default=os.environ.get("JSONL_PATH", 
                            "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/Video-3D-LLM_data/processed/scanqa_train_llava_style.json"),
                        help="JSON/JSONL 数据文件路径")
    parser.add_argument("--num_samples", type=int, 
                        default=int(os.environ.get("NUM_SAMPLES", "5")),
                        help="遍历的样本数")
    parser.add_argument("--strategy", type=str,
                        default=os.environ.get("STRATEGY", "uniform"),
                        help="帧采样策略")
    parser.add_argument("--num_used_data", type=int, default=100,
                        help="从数据文件中使用的数据条数")
    parser.add_argument("--generative_feature_source", type=str, default="none",
                        help="生成特征来源: offline / none")
    args = parser.parse_args()

    print("=" * 60)
    print("  VideoLLM3D Dataset Debug")
    print("=" * 60)
    print(f"  data_dir:       {args.data_dir}")
    print(f"  annotation_dir: {args.annotation_dir}")
    print(f"  jsonl_path:     {args.jsonl_path}")
    print(f"  num_samples:    {args.num_samples}")
    print(f"  strategy:       {args.strategy}")
    print(f"  num_used_data:  {args.num_used_data}")
    print(f"  gen_feat_src:   {args.generative_feature_source}")
    print("=" * 60)

    # ── Mock vit_transform ──
    # 模拟 Bagel 的 vit_transform：输入 PIL 图片列表，输出 (image_tensor, image_grid_thw)
    class MockVitTransform:
        def __init__(self, image_size=384, patch_size=14):
            self.image_size = image_size
            self.patch_size = patch_size
            self.stride = patch_size
        
        def __call__(self, images, img_num=1):
            """
            模拟 vit_transform：
              - 输入: images = [PIL.Image], img_num = int
              - 输出: (image_tensor, image_grid_thw)
                - image_tensor: (num_patches * 4, hidden_dim) 模拟
                - image_grid_thw: [(t, h, w)]
            """
            img = images[0]
            # resize 到 image_size
            img = img.resize((self.image_size, self.image_size))
            img_array = np.array(img).astype(np.float32) / 255.0
            
            h_patches = self.image_size // self.patch_size
            w_patches = self.image_size // self.patch_size
            num_patches = h_patches * w_patches
            
            # 模拟输出：(num_patches * 4, 3) —— 实际 Bagel 中是 (num_patches * 4, hidden_dim)
            image_tensor = torch.randn(num_patches * 4, 3)
            image_grid_thw = [torch.tensor([1, h_patches, w_patches])]
            
            return image_tensor, image_grid_thw

    # ── Mock tokenizer ──
    class MockTokenizer:
        """模拟 tokenizer：简单按空格分词，返回整数 ID 列表。"""
        def __init__(self):
            self.vocab = {}
            self._next_id = 100
        
        def encode(self, text):
            tokens = text.split()
            ids = []
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = self._next_id
                    self._next_id += 1
                ids.append(self.vocab[t])
            return ids

    # ── Mock frame_sampler ──
    class MockFrameSampler:
        """Fallback 帧采样器（不会被 3D 视频使用）。"""
        def __call__(self, file_name):
            return []

    # ── 构建数据集 ──
    vit_transform = MockVitTransform()
    tokenizer = MockTokenizer()
    frame_sampler = MockFrameSampler()

    print("\n[1/3] 初始化 VideoLLM3DIterableDataset ...")
    dataset = VideoLLM3DIterableDataset(
        dataset_name="videollm3d_debug",
        vit_transform=vit_transform,
        tokenizer=tokenizer,
        frame_sampler=frame_sampler,
        jsonl_path_list=[args.jsonl_path],
        data_dir_list=[""],  # image_dir 对 3D 视频不使用
        num_used_data=[args.num_used_data],
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        shuffle_seed=42,
        task_grouped_shuffle=True,
        batch_size=4,
        # ── VideoProcessor 扁平参数 ──
        video_folder=args.data_dir,
        annotation_dir=args.annotation_dir,
        metadata_dir=os.path.join(args.data_dir, 'metadata'),
        voxel_size=0.1,
        min_xyz_range=[-15, -15, -5],
        max_xyz_range=[15, 15, 5],
        frame_sampling_strategy=args.strategy,
        val_box_type='pred',
        force_sample=False,
        frames_upbound=0,
        generative_model_id=None,
        generative_feature_source=args.generative_feature_source,
        add_spatial_instruction=False,
    )
    print(f"  数据集大小: {len(dataset.data_paths)} 条")
    print(f"  每 rank 数据: {len(dataset.data_paths_per_rank)} 条")

    print(f"\n[2/3] 开始遍历前 {args.num_samples} 个样本 ...")
    count = 0
    for sample in dataset:
        count += 1
        print(f"\n{'─' * 50}")
        print(f"  样本 #{count}")
        print(f"  num_tokens:          {sample['num_tokens']}")
        print(f"  image_tensor_list:   {len(sample['image_tensor_list'])} 张图片")
        if sample['image_tensor_list']:
            print(f"    第一张 shape:      {sample['image_tensor_list'][0].shape}")
        print(f"  text_ids_list:       {len(sample['text_ids_list'])} 段文本")
        for i, ids in enumerate(sample['text_ids_list']):
            print(f"    段 {i}: {len(ids)} tokens")
        print(f"  image_grid_thw_list: {len(sample['image_grid_thw_list'])} 个 grid")
        print(f"  sequence_plan:       {len(sample['sequence_plan'])} 步")
        for i, plan in enumerate(sample['sequence_plan']):
            print(f"    步 {i}: type={plan['type']}, loss={plan['loss']}")
        
        if 'video_dict' in sample:
            vd = sample['video_dict']
            print(f"  video_dict:")
            print(f"    video_id:          {vd.get('video_id', 'N/A')}")
            if vd.get('world_coords') is not None:
                print(f"    world_coords:      {vd['world_coords'].shape}")
            if vd.get('boundry') is not None:
                print(f"    boundry:           {vd['boundry']}")
            if vd.get('objects') is not None:
                print(f"    objects:           {vd['objects'].shape}")
            else:
                print(f"    objects:           None")
            if vd.get('generative_feature') is not None:
                print(f"    generative_feat:   {vd['generative_feature'].shape}")
            else:
                print(f"    generative_feat:   None")
            print(f"    box_input:         {vd.get('box_input', 'N/A')}")
        
        if 'box_label' in sample:
            print(f"  box_label:           {sample['box_label']}")

        if count >= args.num_samples:
            break

    print(f"\n{'═' * 60}")
    print(f"[3/3] 完成！成功遍历 {count} 个样本。")
    if count == 0:
        print("  ⚠️  没有成功产出任何样本，请检查：")
        print("     1. 数据文件路径是否正确")
        print("     2. embodiedscan pkl 文件是否存在")
        print("     3. 视频帧文件（.jpg/.png/.txt）是否存在")
    print("=" * 60)


if __name__ == "__main__":
    main()
