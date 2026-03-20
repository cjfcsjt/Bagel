# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, MaskedReconIterableDataset
from .recon_dataset import SftJSONLIterableReconDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.recon_then_und_dataset import ReconthenUndIterableDataset
from .interleave_datasets.und_dataset import UndIterableDataset
from .interleave_datasets.recon_dataset_parquet import ReconParquetIterableDataset

DATASET_REGISTRY = {
    # 'recon': SftJSONLIterableReconDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'recon_then_und': ReconthenUndIterableDataset,
    'recon_parquet': ReconParquetIterableDataset,
    'recon': MaskedReconIterableDataset,
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
            'data_dir': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube_raw_qa_qwen_sft/mindcube',
            'parquet_info_path': '/apdcephfs_303747097/share_303747097/jingfanchen/data/sft/unified_parquets_mindcube_raw_qa_qwen_sft/parquet_info.json',
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