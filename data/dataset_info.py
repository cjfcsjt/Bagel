# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .recon_dataset import SftJSONLIterableReconDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.recon_then_und_dataset import ReconthenUndIterableDataset


DATASET_REGISTRY = {
    'recon': SftJSONLIterableReconDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'recon_then_und': ReconthenUndIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    'recon': {
        'blendedmvs': {
            'data_dir': '/data/spatial_data/data/blendedmvs', # path of the parquet files
            'jsonl_path': '/data/spatial_data/data/blendedmvs/processed/blendmvs_scenes.jsonl', # path of the jsonl file
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
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
			'data_dir': '/data/spatial_data/reason_data/bagel_converted/vlm/images',
			'jsonl_path': '/data/spatial_data/reason_data/bagel_converted/vlm/llava_ov_si.jsonl',
			'num_total_samples': 50000
		},
    },
	'recon_then_und':{
		'spatial_mix': {
			'data_dir': "/data/spatial_data/reason_data/unified_parquets/spar_no_3d_annotation/",
			'num_files': 17,
			'num_total_samples': 7901248,
			"parquet_info_path": '/data/spatial_data/reason_data/unified_parquets/parquet_info.json', # information of the parquet files
		},
	},
}