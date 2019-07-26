#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
# --datasetRoot /home/galen/deepLearningCode/PoseEstimation/DenseFusion/datasets/linemod/Linemod_preprocessed

python ./tools/trainNet.py --dataset linemod\
  --datasetRoot /home/galen/deepLearning/poseEstimation/DenseFusion/datasets/linemod/Linemod_preprocessed\
  --startEpoch 141\
  --resumePosenet pose_model_133_0.01387673569878028.pth
  # --resume_refinenet pose_refine_model_current.pth\
  
