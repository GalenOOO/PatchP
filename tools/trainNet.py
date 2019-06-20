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
parser.add_argument('--w', default=0.015, help='loss calculation hyperparameter')

parser.add_argument('--decayMargin', default=0.016, help='margin to decay lr ')
parser.add_argument('--addNoise',type=bool, default=True, help='whether adding random noise to the training data or not')
parser.add_argument('--noiseTrans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=3, help='max number of epochs to train')
parser.add_argument('--resumePosenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--startEpoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

def main():
    opt.manualSeed = random.randint(1,10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed) #torch.manual_seed可以保证在种子值不变的情况下，每次torch.rand产生的结果相同
    # torch.cuda.manual_seed(opt.manualSeed)   torch.cuda.manual_seed_all(seed)  
    # 对于可重复的实验，有必要为任何使用随机数生成的进行随机种子设置。注意，cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用。

    # 根据数据集配置物体种类，用于预测位姿的点数，每个epoch训练的次数，模型保存的位置，log保存的位置
    if opt.dataset == 'linemod':
        opt.numObjects = 13
        opt.numPoints = 500
        opt.repeatEpoch = 20
        opt.modelFolder = 'trainedModels/'
        opt.logFolder = 'experimentResult/logs/'
    else:
        print('Unknown dataset')
        return


    # 定义网络，如果有训练过的模型，可加载；   确定网络优化方法    
    estimator = poseNet(opt.numPoints, opt.numObjects)
    estimator.cuda() # or estimator.to('cuda')
    if opt.resumePosenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.modelOutFolder,opt.resumePosenet)))
    
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    opt.decay_start = False
    opt.refine_start = False
    # 加载训练数据和测试数据
    if opt.dataset == 'linemod':
        dataset = PoseDataset('train', opt.numPoints, opt.addNoise, opt.datasetRoot, opt.noiseTrans, opt.refine_start)
        test_dataset = PoseDataset('test', opt.numPoints, False, opt.datasetRoot, 0.0, opt.refine_start)
    dataLoader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=True, num_workers=opt.workers)
    testdataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    opt.symList = dataset.get_sym_list()
    opt.numPointsMesh = dataset.get_num_points_mesh()
    print('>>>>>>--------Dataset Loaded!---------<<<<<<<\nlength of the training set:{0}\nlength of the testing set:{1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.numPointsMesh, opt.symList))
    
    # 定义网络训练的误差
    poseNetLoss = Loss(opt.numPointsMesh,opt.symList)
    best_distance = np.Inf

    if opt.startEpoch == 1:
        for log in os.listdir(opt.logFolder):
            os.remove(os.path.join(opt.logFolder,log))
    st_time = time.time()


    for epoch in range(opt.startEpoch, opt.nepoch):
        # ------------------模型训练阶段------------------
        logger = setup_logger('epoch%d'% epoch, os.path.join(opt.logFolder,'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-st_time)) + ',' + 'Training started'))
        train_count = 0
        train_dis_sum = 0.0
        estimator.train()
        optimizer.zero_grad()
        for rep in range(opt.repeatEpoch):
            for i,data in enumerate(dataLoader,0):
                img, cloud, choose, tarPoints, modelPoints, idx = data
                img = Variable(img).cuda()
                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                tarPoints = Variable(tarPoints).cuda()
                modelPoints = Variable(modelPoints).cuda()
                idx = Variable(idx).cuda()
                pred_r, pred_t, pred_c, colorEmb = estimator(img,cloud,choose,idx)
                loss , dis, newCloud, newTarPoints = poseNetLoss(pred_r,pred_t,pred_c,tarPoints,modelPoints,idx,cloud,opt.w,opt.refine_start)

                loss.backward()

                train_dis_sum += dis.item()
                train_count +=1

                if train_count % opt.batchSize == 0:
                    logger.info('Train time {0} Epoch {1} Batch{2} Frame{3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)),epoch,int(train_count/opt.batchSize), train_count, train_dis_sum / opt.batchSize))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_sum = 0
                
                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.modelFolder))
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        # ------------------模型测试阶段------------------
        logger = setup_logger('epoch%d_test'%epoch, os.path.join(opt.logFolder,'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        for j,data in enumerate(testdataLoader,0):
            img, cloud, choose, tarPoints, modelPoints, idx = data
            img = Variable(img).cuda()
            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            tarPoints = Variable(tarPoints).cuda()
            modelPoints = Variable(modelPoints).cuda()
            idx = Variable(idx).cuda()
            pred_r, pred_t, pred_c, colorEmb = estimator(img,cloud,choose,idx)
            loss , dis, newCloud, newTarPoints = poseNetLoss(pred_r,pred_t,pred_c,tarPoints,modelPoints,idx,cloud,opt.w,opt.refine_start)

            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
            test_dis += dis.item()
            test_count += 1
        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))

        # ------------------模型评估阶段-------------------
        if test_dis < best_distance:
            best_distance = test_dis
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.modelFolder,epoch,test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
        
        if best_distance < opt.decayMargin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_dr
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

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

#%%
