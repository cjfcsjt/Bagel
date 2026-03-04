#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set the necessary environment variables
export IMAGE_FOLDER="examples/dl3dv/"
export MODEL_PATH="InternRobotics/G2VLM-2B-MoT"
export SAVE_PATH="results/arkitscenes_results.ply"

# Run the evaluation script
python eval/point/scannet/eval_track.py --image_folder $IMAGE_FOLDER --model_path $MODEL_PATH --save_path $SAVE_PATH