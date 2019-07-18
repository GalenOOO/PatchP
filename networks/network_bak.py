import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=[1, 2, 3, 6], feature_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=[1, 2, 3, 6], feature_size=512, deep_features_size=256, backend='resnet34'),
}

class modifiedResNet(nn.Module):
    def __init__(self,usegpu=True):
        super(modifiedResNet,self).__init__()
        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)
    
    def forward(self, inputImg):
        colorEmb = self.model(inputImg)
        return colorEmb

class featureFusionNet(nn.Module):
    def __init__(self,num_points):
        super(featureFusionNet,self).__init__()
        self.num_points = num_points
        # 这里用的一维卷积，核的大小为1，相当于只做了通道数的变换！！！！！这里考虑将核变大[改进]
        self.cloudConv1 = nn.Conv1d(3,64,1)
        self.cloudConv2 = nn.Conv1d(64,128,1)

        self.colorConv1 = nn.Conv1d(32,64,1)
        self.colorConv2 = nn.Conv1d(64,128,1)

        self.featConv1 = nn.Conv1d(256,512,1)
        self.featConv2 = nn.Conv1d(512,1024,1) 

        self.getGlobalFeat = nn.AvgPool1d(num_points)
        
    def forward(self,cloud,colorEmb):
        cloudFeat1 = F.relu(self.cloudConv1(cloud))
        colorFeat1 = F.relu(self.colorConv1(colorEmb))
        catFeat1 = torch.cat((cloudFeat1,colorFeat1), dim=1) #64+64=128dim

        cloudFeat2 = F.relu(self.cloudConv2(cloudFeat1))
        colorFeat2 = F.relu(self.colorConv2(colorFeat1))
        catFeat2 = torch.cat((cloudFeat2,colorFeat2), dim=1) #128+128=256dim

        fusionFeat = F.relu(self.featConv1(catFeat2))
        fusionFeat = F.relu(self.featConv2(fusionFeat))

        globalFeat = self.getGlobalFeat(fusionFeat) #globalFeat: bs*1024*1
        globalFeat = globalFeat.view(-1,1024,1).repeat(1,1,self.num_points)
        # 返回结果是bs×1408×num_points维的变量
        return torch.cat([catFeat1,catFeat2,globalFeat],dim=1)#128 + 256 + 1024

class featureExtraction(nn.Module):
    def __init__(self,pooledImgSize):
        super(featureExtraction,self).__init__()
        self.pooledImgSize = pooledImgSize
        self.Conv1 = nn.Conv2d(64,128,1)
        self.Conv2 = nn.Conv2d(128,256,1)
        self.Conv3 = nn.Conv2d(256,512,1)
        self.Conv4 = nn.Conv2d(512,1024,1)
        self.getGlobalFeat = nn.AvgPool2d(pooledImgSize)
        
    def forward(self,colorEmb):
        colorFeat = F.relu(self.Conv1(colorEmb)) 
        colorFeat = F.relu(self.Conv2(colorFeat))
        colorFeat = F.relu(self.Conv3(colorFeat))
        colorFeat = F.relu(self.Conv4(colorFeat)) # 1024dim * (pooledImgSize * pooledImgSize)

        globalFeat = self.getGlobalFeat(colorFeat) #globalFeat: bs*1024*1
        globalFeat = globalFeat.view(-1,1024)
        # globalFeat = globalFeat.view(-1,1024,1,1).repeat(1,1,self.pooledImgSize,self.pooledImgSize)
        # return torch.cat([colorFeat,globalFeat],dim=1)#1024 + 1024 = 2048dim
        return colorFeat,globalFeat #1024  1024 


class colorNet(nn.Module):
    def __init__(self):
        super(colorNet,self).__init__()
        self.layer1 = nn.Conv1d(3,32,1)
        self.layer2 = nn.Conv1d(32,64,1)
        self.layer3 = nn.Conv1d(64,128,1)
        self.layer4 = nn.Conv1d(128,256,1)
        self.layer5 = nn.Conv1d(256,512,1)   
    def forward(self, colorImg):
        colorFeat = F.relu(self.layer1(colorImg))
        colorFeat = F.relu(self.layer2(colorFeat))
        colorFeat = F.relu(self.layer3(colorFeat))
        colorFeat1 = F.relu(self.layer4(colorFeat))
        colorFeat2 = F.relu(self.layer5(colorFeat1))
        return colorFeat1,colorFeat2

class cloudNet(nn.Module):
    def __init__(self):
        super(cloudNet,self).__init__()
        self.layer1 = nn.Conv1d(3,32,1)
        self.layer2 = nn.Conv1d(32,64,1)        
        self.layer3 = nn.Conv1d(64,128,1)
        self.layer4 = nn.Conv1d(128,256,1)    
    def forward(self, cloud):
        cloudFeat = F.relu(self.layer1(cloud))
        cloudFeat = F.relu(self.layer2(cloudFeat))
        cloudFeat1 = F.relu(self.layer3(cloudFeat))
        cloudFeat2 = F.relu(self.layer4(cloudFeat1))
        return cloudFeat1,cloudFeat2

class featFusion(nn.Module):
    def __init__(self,num_points):
        super(featFusion,self).__init__()
        self.num_points = num_points
        # 这里用的一维卷积，核的大小为1，相当于只做了通道数的变换！！！！！这里考虑将核变大[改进]
        self.featConv1 = nn.Conv1d(384,512,1) 
        self.featConv2 = nn.Conv1d(768,1024,1)
        
        self.getGlobalFeat = nn.AvgPool1d(num_points)
        
    def forward(self,cloudFeat1,cloudFeat2,colorFeat1,colorFeat2): 
        catFeat1 = torch.cat((cloudFeat1,colorFeat1), dim=1) #128+256=384dim
        catFeat2 = torch.cat((cloudFeat2,colorFeat2), dim=1) #256+512=768dim

        fusionFeat1 = F.relu(self.featConv1(catFeat1))
        fusionFeat2 = F.relu(self.featConv2(catFeat2))

        fusionFeat = torch.cat((fusionFeat1, fusionFeat2), dim=1) #512+1024 = 1536

        globalFeat = self.getGlobalFeat(fusionFeat) #globalFeat: bs*1024*1
        globalFeat = globalFeat.view(-1,1536)
        # 返回结果是bs×1408×num_points维的变量
        return fusionFeat,globalFeat
        # return torch.cat([fusionFeat,globalFeat],dim=1)#128 + 256 + 1024

class poseNet(nn.Module): #必须继承nn.Module
    def __init__(self,num_points, num_obj): 
        super(poseNet, self).__init__() #需要调用父类的构造方法
        self.num_points = num_points
        self.num_obj = num_obj
        # 网络结构定义
        self.bodyCNN = modifiedResNet()
        self.colorNet = colorNet()
        self.cloudNet = cloudNet()
        self.fusionNet = featFusion(num_points)
        self.feat = featureExtraction(num_points)

        # ------------global feature start----------------
        self.Rconv1 = nn.Linear(1536,512)
        self.Tconv1 = nn.Linear(1536,512)
        self.Cconv1 = nn.Linear(1536,512)

        self.Rconv2 = nn.Linear(512,128)
        self.Tconv2 = nn.Linear(512,128)
        self.Cconv2 = nn.Linear(512,128)

        # self.Rconv3 = nn.Linear(128,num_obj*4)
        # self.tconv3 = nn.Linear(128,num_obj*3)
        # ------------global feature end----------------
        self.conv1_r = nn.Conv1d(1536,512,3)
        self.conv1_t = nn.Conv1d(1536,512,3)
        self.conv1_c = nn.Conv1d(1536,512,3)
        #----------以下这一段略作修改，只使用patchFeat，不使用globalFeat
        self.conv2_r = nn.Conv1d(512,256,3)
        self.conv2_t = nn.Conv1d(512,256,3)
        self.conv2_c = nn.Conv1d(512,256,3)
        #----------------------
        self.conv3_r = nn.Conv1d(256,128,3)
        self.conv3_t = nn.Conv1d(256,128,3)
        self.conv3_c = nn.Conv1d(256,128,3)

        self.conv4_r = nn.Conv1d(256,num_obj*4,1) #四元数
        self.conv4_t = nn.Conv1d(256,num_obj*3,1)
        self.conv4_c = nn.Conv1d(256,num_obj*1,1)

    def forward(self, colorImg,cloud, obj):
        cloud = cloud.transpose(2,1).contiguous() 
        colorImg = colorImg.transpose(2,1).contiguous() 
        
        colorFeat1, colorFeat2 = self.colorNet(colorImg) #colorFeat的di应该是32。大小和输入的img一样
        cloudFeat1, cloudFeat2 = self.cloudNet(cloud)
        fusionFeat,globalFeat = self.fusionNet(cloudFeat1, cloudFeat2, colorFeat1, colorFeat2)
        #------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---------------------
        # 融合globalFeat 和 patchFeat
        rx_g = F.relu(self.Rconv1(globalFeat)) # dim 1536 -> 512
        tx_g = F.relu(self.Tconv1(globalFeat))
        cx_g = F.relu(self.Cconv1(globalFeat))

        rx_g = F.relu(self.Rconv2(rx_g)) # dim 512 -> 128
        tx_g = F.relu(self.Tconv2(tx_g))
        cx_g = F.relu(self.Cconv2(cx_g))

        rx = F.relu(self.conv1_r(fusionFeat)) #dim = 512
        tx = F.relu(self.conv1_t(fusionFeat)) #dim = 512
        cx = F.relu(self.conv1_c(fusionFeat)) #dim = 512
        
        rx = F.relu(self.conv2_r(rx)) #dim = 256
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx)) #dim = 256
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        bs, dimensions, numPatchs =  cx.size()
        rx_g = rx_g.view(-1,128,1).repeat(1,1,numPatchs) # dim = 128
        tx_g = tx_g.view(-1,128,1).repeat(1,1,numPatchs)
        cx_g = cx_g.view(-1,128,1).repeat(1,1,numPatchs)


        rxFusion = torch.cat([rx,rx_g],dim=1)  #dim = 128+128 = 256
        txFusion = torch.cat([tx,tx_g],dim=1)
        cxFusion = torch.cat([cx,cx_g],dim=1)

        #------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---------------------        

        rx = self.conv4_r(rxFusion).view(bs,self.num_obj,4,-1)
        tx = self.conv4_t(txFusion).view(bs,self.num_obj,3,-1)
        cx = torch.sigmoid(self.conv4_c(cxFusion)).view(bs,self.num_obj,1,-1)
        # 上面代码是对每一个融合后特征点进行了位姿估计

        temp = 0
        outRx = torch.index_select(rx[temp],0,obj[temp]) #找出属于目标种类的预测结果1×4×num_cloudPoints
        outTx = torch.index_select(tx[temp],0,obj[temp]) #这里应该是bs=1，所以可以直接tx[temp]
        outCx = torch.index_select(cx[temp],0,obj[temp])
        # torch.index_select(input, dim, index, out=None) → Tensor
        # 保留input的dim维，obj[temp]指定的数

        outRx = outRx.contiguous().transpose(2,1).contiguous()
        outTx = outTx.contiguous().transpose(2,1).contiguous()
        outCx = outCx.contiguous().transpose(2,1).contiguous()

        return outRx, outTx, outCx, fusionFeat.detach() #detach()作用是不让变量求导，仍与colorEmb指向同一tensor
        # torch.Size([1, 500, 4])
        # torch.Size([1, 500, 3])
        # torch.Size([1, 500, 1])
        # torch.Size([1, 32, 500])


class poseRefineNetFeat(nn.Module):
    def __init__(self,num_points):
        super(poseRefineNetFeat,self).__init__()
        self.num_points = num_points
        # 这里用的一维卷积，核的大小为1，相当于只做了通道数的变换
        self.cloudConv1 = nn.Conv1d(3,64,1)
        self.cloudConv2 = nn.Conv1d(64,128,1)

        self.colorConv1 = nn.Conv1d(32,64,1)
        self.colorConv2 = nn.Conv1d(64,128,1)

        self.featConv1 = nn.Conv1d(384,512,1)
        self.featConv2 = nn.Conv1d(512,1024,1) 

        self.getGlobalFeat = nn.AvgPool1d(num_points)
        
    def forward(self,cloud,colorEmb):
        cloudFeat1 = F.relu(self.cloudConv1(cloud))
        colorFeat1 = F.relu(self.colorConv1(colorEmb))
        catFeat1 = torch.cat((cloudFeat1,cloudFeat1), dim=1) #64+64=128dim

        cloudFeat2 = F.relu(self.cloudConv2(cloudFeat1))
        colorFeat2 = F.relu(self.colorConv2(colorFeat1))
        catFeat2 = torch.cat((cloudFeat2,colorFeat2), dim=1) #128+128=256dim

        fusionFeat = torch.cat([catFeat1, catFeat2], dim=1)

        fusionFeat = F.relu(self.featConv1(fusionFeat))
        fusionFeat = F.relu(self.featConv2(fusionFeat))

        globalFeat = self.getGlobalFeat(fusionFeat) #globalFeat: bs*1024*1
        globalFeat = globalFeat.view(-1,1024)
        return globalFeat



class poseRefineNet(nn.Module):
    def __init__(self,num_points,num_obj):
        super(poseRefineNet,self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.feat = poseRefineNetFeat(num_points)

        self.Rconv1 = nn.Linear(1024,512)
        self.tconv1 = nn.Linear(1024,512)

        self.Rconv2 = nn.Linear(512,128)
        self.tconv2 = nn.Linear(512,128)

        self.Rconv3 = nn.Linear(128,num_obj*4)
        self.tconv3 = nn.Linear(128,num_obj*3)

    def forward(self, cloudTrans, imgEmb, obj):
        bs = cloudTrans.size()[0]
        
        cloudTrans = cloudTrans.transpose(2,1).contiguous()

        globalFeature = self.feat(cloudTrans,imgEmb)

        rx = F.relu(self.Rconv1(globalFeature))
        tx = F.relu(self.tconv1(globalFeature))

        rx = F.relu(self.Rconv2(rx))
        tx = F.relu(self.tconv2(tx))

        rx = self.Rconv3(rx).view(bs,self.num_obj,4)
        tx = self.tconv3(tx).view(bs,self.num_obj,3)

        tmp = 0
        out_rx = torch.index_select(rx[tmp],0,obj[tmp])
        out_tx = torch.index_select(tx[tmp],0,obj[tmp])

        return out_rx,out_tx

