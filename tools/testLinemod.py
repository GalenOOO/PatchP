
import _init_paths
import argparse
import os
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from datasets.dataset import PoseDataset as PoseDataset
from networks.network import poseNet,poseRefineNet

from libs.loss import Loss
from libs.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from tools.utils import setup_logger
from libs.knn.__init__ import KNearestNeighbor
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--datasetRoot', type=str, default='', help='dataset root dir')
parser.add_argument('--model', type=str, default='', help='resume poseNet model')

opt = parser.parse_args()

numObjects = 13
objList = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
numPoints = 576
pooledImgSize = 48
batchSize = 1
datasetConfigDir = '/home/galen/deepLearning/poseEstimation/DenseFusion/datasets/linemod/Linemod_preprocessed/models/'
output_result_dir = 'experimentResult/eval_results/linemod'
knn = KNearestNeighbor(1)

estimator = poseNet(numPoints,numObjects)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

testDataset = PoseDataset('eval',numPoints,False,opt.datasetRoot,0.0,True)
testDataLoader = torch.utils.data.DataLoader(testDataset,batch_size=1,shuffle=False, num_workers=10)

symList = testDataset.get_sym_list()
numPointsMesh = testDataset.get_num_points_mesh()
poseNetLoss = Loss(numPointsMesh,symList)

diameter = []
meta_file = open('{0}/models_info.yml'.format(datasetConfigDir), 'r')
meta = yaml.load(meta_file)
kn = 0.1 # ADD 参数设置
for obj in objList:
    diameter.append(meta[obj]['diameter']/1000.0 * kn)
print(diameter)

successCount = [0 for i in range(numObjects)]
numCount = [0 for i in range(numObjects)]
cnt = 0
for i,data in enumerate(testDataLoader, 0):
    if len(data) != 7:
        print(i,len(data))
        continue
    img, cloud, pointIndex ,tarPoints, modelPoints, idx, ori_img = data
                
    
    cnt += 1
    if cnt%100 != 0:
        continue
    if len(cloud.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        continue
    img = Variable(img).cuda()
    cloud = Variable(cloud).cuda()
    pointIndex = Variable(pointIndex).cuda()
    tarPoints = Variable(tarPoints).cuda()
    modelPoints = Variable(modelPoints).cuda()
    idx = Variable(idx).cuda()
    ori_img = np.array(ori_img)

    pred_r, pred_t, pred_c, colorEmb = estimator(img,cloud,pointIndex,idx)

    bs, num_p,_ = pred_c.size()
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, num_p, 1))
    pred_c = pred_c.view(batchSize, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(batchSize*num_p,1,3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    # my_t = (cloud.view(batchSize * num_p, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_t = pred_t[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    modelPoints = modelPoints[0].cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3,:3]
    pred = np.dot(modelPoints,my_r.T) + my_t

    cam_cx = 325.26110
    cam_cy = 242.04899
    cam_fx = 572.41140
    cam_fy = 573.57043
    d = pred[:,2] / 1.0
    rows = pred[:,0] * cam_fx / pred[:,2] + cam_cx
    cols = pred[:,1] * cam_fy / pred[:,2] + cam_cy

    tarPoints = tarPoints[0].cpu().detach().numpy()

    if idx[0].item() in symList:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        tarPoints = torch.from_numpy(tarPoints.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(tarPoints.unsqueeze(0), pred.unsqueeze(0))
        tarPoints = torch.index_select(tarPoints, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - tarPoints.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - tarPoints, axis=1))

    if dis < diameter[idx[0].item()]:
        successCount[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        plt.imshow(ori_img[0])
        plt.plot(rows,cols,'r,')
        plt.show()
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        plt.imshow(ori_img[0])
        plt.plot(rows,cols,'b,')
        plt.show()
    numCount[idx[0].item()] += 1

    if cnt > 10000:
        break





