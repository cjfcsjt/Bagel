
cd /data/spatial_data/Bagel/eval/spatial_reason/ost/OST-Bench
python QwenVL_baseline.py --rank_index 0 --rank_num 4 --model_path Qwen/Qwen2.5-VL-3B-Instruct --save_root /data/spatial_data/Bagel/eval/spatial_reason/ost/OST-Bench/ --anno_json_path /data/spatial_data/reason_data/OST-Bench/OST_bench.json --image_root /data/spatial_data/reason_data/OST-Bench/image_upload/