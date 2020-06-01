#!/bin/bash
source setup.sh
python ../models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=$(pwd)/lisa/experiments/training/faster_rcnn_lisa.config --trained_checkpoint_prefix=$(pwd)/lisa/experiments/training/model.ckpt-21103 --output_directory=$(pwd)/lisa/experiments/exported_model