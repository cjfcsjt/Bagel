# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .recon_dataset import SftJSONLIterableReconDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    'recon_pretrain': SftJSONLIterableReconDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    'recon_pretrain': {
        'recon': {
            'data_dir': '/data/spatial_data/bagel_example/recon', # path of the parquet files
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
			'data_dir': '/data/spatial_data/bagel_example/vlm/images',
			'jsonl_path': '/data/spatial_data/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}