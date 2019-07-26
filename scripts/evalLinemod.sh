#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/evalLinemod.py --datasetRoot /home/galen/deepLearning/poseEstimation/DenseFusion/datasets/linemod/Linemod_preprocessed\
  --model trainedModels/pose_model_191_0.013623646898686602.pth
  # --resume_refinenet pose_refine_model_current.pth\
  