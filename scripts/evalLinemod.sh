#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/evalLinemod.py --datasetRoot /home/galen/deepLearningCode/PoseEstimation/DenseFusion/datasets/linemod/Linemod_preprocessed\
  --model trainedModels/pose_model_17_0.01738924132724513.pth
  # --resume_refinenet pose_refine_model_current.pth\
  