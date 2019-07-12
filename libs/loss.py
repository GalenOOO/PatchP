from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
from libs.knn.__init__ import KNearestNeighbor

def loss_calculation(pred_r,pred_t,pred_c,targetCloud, modelPoints,idx,cloud,w,refine,num_pt_mesh,symList):
    knn = KNearestNeighbor(1)
    bs, num_p,_ = pred_c.size()

    pred_r = pred_r / (torch.norm(pred_r,dim=2).view(bs,num_p,1))
    # 将四元数转换成旋转矩阵 
    rMat = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    ori_rMat = rMat
    pred_t = pred_t.contiguous().view(bs*num_p,1,3)
    ori_tMat = pred_t

    modelPoints = modelPoints.view(bs,1,num_pt_mesh,3).repeat(1,num_p,1,1).view(bs*num_p, num_pt_mesh,3)
    targetCloud = targetCloud.view(bs,1,num_pt_mesh,3).repeat(1,num_p,1,1).view(bs*num_p, num_pt_mesh,3)
    ori_tarCloud = targetCloud
    
    cloud = cloud.contiguous().view(-1,1,3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    rMat = rMat.contiguous().transpose(2,1).contiguous() #这里为什么转置？正常来讲，是rMat×point，而在下面计算预测值时，由于有很多point，所以采用了point×rMat
    # predCloud = torch.add(torch.bmm(modelPoints,rMat),cloud+pred_t) #####  训练的t就是对每个点的补偿值，所以加上cloud（可考虑将Cloud删除进行测试）[改进]
    predCloud = torch.add(torch.bmm(modelPoints,rMat),pred_t) #####  训练的t就是对每个点的补偿值，所以加上cloud（Cloud已删除进行测试）[改进]
    # torch.bmm(batch1, batch2, out=None) → Tensor 
    # If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.

    if not refine:
        if idx[0].item() in symList:
            targetCloud = targetCloud[0].transpose(1,0).contiguous().view(3,-1)
            predCloud = predCloud.permute(2,0,1).contiguous().view(3,-1)
            inds = knn(targetCloud.unsqueeze(0),predCloud.unsqueeze(0))
            targetCloud = torch.index_select(targetCloud, 1, inds.view(-1).detach() -1)
            targetCloud = targetCloud.view(3, bs*num_p, num_pt_mesh).permute(1,2,0).contiguous()
            predCloud = predCloud.view(3, bs*num_p, num_pt_mesh).permute(1,2,0).contiguous()
    
    dis = torch.mean(torch.norm((predCloud-targetCloud), dim=2), dim=1) # [bs*num_p]
    loss = torch.mean((dis*pred_c - w*torch.log(pred_c)), dim=0)

    pred_c = pred_c.view(bs,num_p)
    how_max, which_max = torch.max(pred_c,1)
    dis = dis.view(bs,num_p)

    t = ori_tMat[which_max[0]] #+ cloud[which_max[0]] #对应前面t是每个点的补偿值 [1,3]
    cloud = cloud.view(1,-1,3)
    _ , numPoints, _ = cloud.size() 

    ori_rMat = ori_rMat[which_max[0]].view(1,3,3).contiguous() #[1,3,3]
    ori_tMat = t.repeat(numPoints,1).contiguous().view(1,numPoints,3)
    # newPoints = torch.bmm((cloud-ori_tMat),ori_rMat).contiguous()  #新的点云相当于做了 所预测的位姿的齐次变换的 逆变换（如果预测的位姿完全正确的话，物体坐标系会和相机坐标系重合）
    newPoints = torch.bmm((cloud-ori_tMat),ori_rMat).contiguous()  #新的点云相当于做了 所预测的位姿的齐次变换的 逆变换（如果预测的位姿完全正确的话，物体坐标系会和相机坐标系重合）
    #这里乘ori_rMat,其实是(ori_rMat.T.T) 第一个.T表示逆矩阵，第二个是因为点在左边，变换矩阵在右边，所以变换矩阵要转置一下
    newTarCloud = ori_tarCloud[0].view(1, num_pt_mesh, 3).contiguous()
    ori_tMat = t.repeat(num_pt_mesh,1).contiguous().view(1,num_pt_mesh,3)
    newTarCloud = torch.bmm((newTarCloud - ori_tMat), ori_rMat).contiguous()

    del knn
    return loss,dis[0][which_max[0]],newPoints.detach(), newTarCloud.detach() #返回值loss，confidence最高的点duiyin 

class Loss(_Loss):
    def __init__(self,num_points_mesh,symList):
        super(Loss,self).__init__(True) #why True?
        self.num_pt_mesh = num_points_mesh
        self.symList = symList

    def forward(self, pred_r, pred_t, pred_c, targetCloud, modelPoints, idx, cloud,w,refine):
        return loss_calculation(pred_r,pred_t,pred_c,targetCloud,modelPoints,idx,cloud,w,refine,self.num_pt_mesh,self.symList)

# #%%
# import torch
# a = torch.rand(1,3)
# print(a)
# b,c = a.size()
# print(b,c)
# print(a.size())
#%%
# import torch
# import numpy as np
# numpy_data = np.arange(36).reshape(2,2,9)
# print('numpy_data',numpy_data)
# torch_data = torch.from_numpy(numpy_data)
# b = torch_data.view(4,3,3)
# print(b)
# c = b.transpose(2,1)
# d = b.transpose(1,2)
# print(c)
# print(d)

# #%%
# import torch
# import numpy as np
# from torch.autograd import Variable

# numpy_data = np.arange(12).reshape(3,4).astype(np.float32)
# print('numpy_data',numpy_data)
# torch_data = torch.from_numpy(numpy_data)
# torch_data = Variable(torch_data).cuda()
# print(torch_data)

# res = torch.max(torch_data)
# res2 = torch.max(torch_data,0)
# res3 = torch.max(torch_data,1)
# print(res3)
# # dis = torch.mean(torch.norm(torch_data, dim=2), dim=1) # bs*num_p,1,1
# # print(dis.size())
