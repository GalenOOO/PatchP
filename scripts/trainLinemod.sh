#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/trainNet.py --dataset linemod\
  --datasetRoot /home/galen/deepLearningCode/PoseEstimation/DenseFusion/datasets/linemod/Linemod_preprocessed
#   --startEpoch 3
#   --resumePosenet pose_model_2_0.026566604857959.pth\
  # --resume_refinenet pose_refine_model_current.pth\
  