# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, MaskedReconIterableDataset, MAEMaskedReconIterableDataset
from .recon_dataset import SftJSONLIterableReconDataset
from .vlm_dataset import SftJSONLIterableDataset
from .videollm3d_dataset import VideoLLM3DIterableDataset
from .interleave_datasets.recon_then_und_dataset import ReconthenUndIterableDataset
from .interleave_datasets.und_dataset import UndIterableDataset
from .interleave_datasets.recon_dataset_parquet import ReconParquetIterableDataset

DATASET_REGISTRY = {
    # 'recon': SftJSONLIterableReconDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'videollm3d': VideoLLM3DIterableDataset,
    'recon_then_und': ReconthenUndIterableDataset,
    'recon_parquet': ReconParquetIterableDataset,
    'recon': MaskedReconIterableDataset,
    'mae_recon': MAEMaskedReconIterableDataset,
    'und': UndIterableDataset,
    'recon_for_und': ReconthenUndIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    'recon': {
        're10k': {
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/', # path of the parquet files
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/parquet_info.json'
        },
        'blendedmvs': {
            'data_dir': '/data/spatial_data/data/blendedmvs', # path of the parquet files
            'jsonl_path': '/data/spatial_data/data/blendedmvs/processed/blendmvs_scenes.jsonl', # path of the jsonl file
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'mae_recon': {
        're10k': {
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/', # path of the parquet files
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_re10k/re10k/parquet_info.json'
        },
        'blendedmvs': {
            'data_dir': '/data/spatial_data/data/blendedmvs', # path of the parquet files
            'jsonl_path': '/data/spatial_data/data/blendedmvs/processed/blendmvs_scenes.jsonl', # path of the jsonl file
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'und':{
		'spatial_mix': {
			'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/mindcube",
			'num_files': 1,
			'num_total_samples': 7901248,
			"parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/parquet_info.json', # information of the parquet files
		},
        'sensense_und': {
            'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_und",
            # 'num_files': 2,
            # 'num_total_samples': 800000,
            "parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_und/parquet_info.json', # information of the parquet files
        },
        'mindcube_raw_qa': {
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10k_raw_qa_shuffle_64group/mindcube',
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10k_raw_qa_shuffle_64group/parquet_info.json',
        },
        "spar-234k": {
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_vgllm/vgllm_spar_234k', # path of the parquet files",
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_vgllm/vgllm_spar_234k/parquet_info.json',
        },
        "spar-234k-mindcube_raw_qa": {
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mix_spar234k-mindcube10k_raw_qa/mixed_mc_vgllm', # path of the parquet files",
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mix_spar234k-mindcube10k_raw_qa/mixed_mc_vgllm/parquet_info.json',
        }
    },
    'recon_for_und':{
		'spatial_mix': {
			'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/mindcube",
			'num_files': 1,
			'num_total_samples': 7901248,
			"parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/parquet_info.json', # information of the parquet files
		},
        'sensense_und': {
            'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_und",
            # 'num_files': 2,
            # 'num_total_samples': 800000,
            "parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_und/parquet_info.json', # information of the parquet files
        },
        'mindcube_raw_qa': {
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube_raw_qa_qwen_sft/mindcube',
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube_raw_qa_qwen_sft/parquet_info.json',
        }
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '/data/spatial_data/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": '/data/spatial_data/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/bagel_converted/vlm/images',
			'jsonl_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/bagel_converted/vlm/llava_ov_si.jsonl',
			'num_total_samples': 50000
		},
    },
    'videollm3d': {
        'scanqa': {
            'data_dir': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/',
            'jsonl_path': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/processed/scanqa_train_llava_style.json',
            'num_total_samples': 30000,
        },
        'sqa3d': {
            'data_dir': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/',
            'jsonl_path': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/processed/sqa3d_train_llava_style.json',
            'num_total_samples': 30000,
        },
        'scan2cap': {
            'data_dir': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/',
            'jsonl_path': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/processed/scan2cap_train_llava_style.json',
            'num_total_samples': 30000,
        },
        'scanrefer': {
            'data_dir': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/',
            'jsonl_path': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/processed/scanrefer_vg_train_llava_style.json',
            'num_total_samples': 30000,
        },
        'multi3drefer': {
            'data_dir': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/',
            'jsonl_path': '/mnt/group/jingfanchen/data/sft/Video-3D-LLM_data/processed/multi3drefer_train_llava_style.json',
            'num_total_samples': 30000,
        },
    },
	'recon_then_und':{
		'spatial_mix': {
			'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/mindcube",
			'num_files': 1,
			'num_total_samples': 7901248,
			"parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/parquet_info.json', # information of the parquet files
		},
        'sensense_und': {
            'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_und",
            # 'num_files': 2,
            # 'num_total_samples': 800000,
            "parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_und/parquet_info.json', # information of the parquet files
        }

    },
    'recon_parquet':{
        'spatial_mix': {
			'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/mindcube",
			'num_files': 1,
			'num_total_samples': 7901248,
			"parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube10/parquet_info.json', # information of the parquet files
		},
        'sensense_geo': {
            'data_dir': "/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_geo",
            # 'num_files': 2,
            # 'num_total_samples': 800000,
            "parquet_info_path": '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_sense800k_dual_tmp/sensenova_si_geo/parquet_info.json', # information of the parquet files
        }
    },
}