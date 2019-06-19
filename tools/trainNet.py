import _init_paths
import argparse
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from datasets.dataset import PoseDataset as PoseDataset
from networks.network import poseNet,poseRefineNet
from libs.loss import Loss
from tools.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'linemod', help='dataset name')
parser.add_argument('--datasetRoot', type=str, default = '', help='dataset root dir')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')

parser.add_argument('--batchSize', type=int, default = 8, help='batch size')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_dr', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')

parser.add_argument('--decayMargin', default=0.016, help='margin to decay lr ')
parser.add_argument('--addNoise',type=bool, default=True, help='whether adding random noise to the training data or not')
parser.add_argument('--noiseTrans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=300, help='max number of epochs to train')
parser.add_argument('--resumePosenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--startEpoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

def main():
    opt.manualSeed = random.randint(1,10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed) #torch.manual_seed可以保证在种子值不变的情况下，每次torch.rand产生的结果相同

    if opt.dataset == 'linemod':
        opt.numObjects = 13
        opt.numPoints = 500
        opt.repeatEpoch = 20
        opt.modelOutFolder = 'trainedModels/'
        opt.logOutFolder = 'experimentResult/logs/'
    else:
        print('Unknown dataset')
        return
    
    estimator = poseNet(opt.numPoints, opt.numObjects)
    estimator.cuda() # or estimator.to('cuda')

    if opt.resumePosenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.modelOutFolder,opt.resumePosenet)))
    
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)



    opt.refine_start = False
    if opt.dataset == 'linemod':
        dataset = PoseDataset('train', opt.numPoints, opt.addNoise, opt.datasetRoot, opt.noiseTrans, opt.refine_start)
        test_dataset = PoseDataset('test', opt.numPoints, False, opt.datasetRoot, 0.0, opt.refine_start)
    dataLoader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=True, num_workers=opt.workers)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.symList = dataset.get_sym_list()
    opt.numPointsMesh = dataset.get_num_points_mesh()

    print('>>>>>>--------Dataset Loaded!---------<<<<<<<\nlength of the training set:{0}\nlength of thr testing set:{1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.numPointsMesh, opt.symList))
    for i, data in enumerate(dataLoader, 0):
        print(i)
        img, points, choose, target, model_points, idx = data
        print(img.size())
        print(points.size())
        print(choose.size())
        print(target.size())
        print(model_points.size())
        # torch.Size([1, 3, 160, 160])
        # torch.Size([1, 500, 3])
        # torch.Size([1, 1, 500])
        # torch.Size([1, 500, 3])
        # torch.Size([1, 500, 3])
        points, choose, img, target, model_points, idx = Variable(points).cuda(), Variable(choose).cuda(), \
           Variable(img).cuda(), Variable(target).cuda(), Variable(model_points).cuda(), Variable(idx).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        # torch.Size([1, 500, 4])
        # torch.Size([1, 500, 3])
        # torch.Size([1, 500, 1])
        # torch.Size([1, 32, 500])
        print(pred_r.size())
        print(pred_t.size())
        print(pred_c.size())
        print(emb.size())
        break



if __name__ == '__main__':
    main()


# #%%
# import torch
# import random
# t = random.randint(1,100)
# print(t)
# random.seed(t)
# torch.manual_seed(t)
# print(t)
# a = torch.rand([1,5])
# b = torch.rand([1,5])
# print(a,b)

# #%%
# seed = 2018
# # torch.manual_seed(seed)
# a=torch.rand([1,5])
# b=torch.rand([1,5])
# print(a,b)
# #%%
