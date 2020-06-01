#!/bin/bash

source setup.sh && python ../models/research/object_detection/legacy/eval.py --logtostderr --pipeline_config_path=$(pwd)/lisa/experiments/training/faster_rcnn_lisa.config --checkpoint_dir=$(pwd)/lisa/experiments/training --eval_dir=$(pwd)/lisa/experiments/evaluation
