#!/bin/bash
source setup.sh 
python predict.py --model lisa/experiments/exported_model/frozen_inference_graph.pb --labels lisa/records/classes.pbtxt --image lisa/aiua120214-1/frameAnnotations-DataLog02142012_001_external_camera.avi_annotations/thruMergeLeft_1331865938.avi_image4.png --num-classes 3