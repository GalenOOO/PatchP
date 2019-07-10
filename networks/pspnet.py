import torch
from torch import nn
from torch.nn import functional as F
import networks.resnet as resnet


class PSPModule(nn.Module):
    def __init__(self,feature_size,outFeatures=1024,sizes=(1,2,3,6)):
        super(PSPModule,self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(feature_size,size) for size in sizes])
        self.convTotal = nn.Conv2d(feature_size * (len(sizes)+1), outFeatures, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self,feature_size,size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size,size)) #输出结果就是size×size大小
        conv = nn.Conv2d(feature_size, feature_size, kernel_size=1, bias=False)
        return nn.Sequential(pool,conv)

    def forward(self,imgFeat):
        h,w = imgFeat.size(2),imgFeat.size(3)
        totalStage = [F.upsample(input=stage(imgFeat),size=(h,w),mode='bilinear') for stage in self.stages] + [imgFeat]
        totalFeat = self.convTotal(torch.cat(totalStage,1))
        return self.relu(totalFeat)

class PSPUpsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PSPUpsample,self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#将大小变为2倍
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1),
            nn.PReLU()
        )
    def forward(self, input):
        return self.conv(input)
        


class PSPNet(nn.Module):
    def __init__(self,sizes=(1,2,3,6), feature_size=2048, deep_features_size=1024, num_classes=21, backend='resnet18',
        pretrained=False):
        super(PSPNet,self).__init__()

        self.resnet = getattr(resnet,backend)(pretrained)
        # Get a named attribute from an object; 
        # getattr(x, 'y') is equivalent to x.y. 
        # When a default argument is given, it is returned when the attribute doesn't exist;
        # without it, an exception is raised in that case.
        self.pspFeatGetFusion = PSPModule(feature_size,1024,sizes)

        self.drop_1 = nn.Dropout2d(p=0.3) #Dropout是将任意一个元素置零，2d将一个1×n置零，3d将1×n×m置零
        
        self.up_1 = PSPUpsample(1024,256)
        self.up_2 = PSPUpsample(256,128)
        self.up_3 = PSPUpsample(128,128)
        self.drop_2 = nn.Dropout2d(p=0.15)
        
        self.final = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=1),
            nn.LogSoftmax()
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size,256),
        #     nn.ReLU(),
        #     nn.Linear(256,num_classes)
        # )
    
    def forward(self,img):
        imgFeat, class_confidence = self.resnet(img)
        pspTotalFeat = self.pspFeatGetFusion(imgFeat)

        pspTotalFeat = self.drop_1(pspTotalFeat)

        pspTotalFeat = self.up_1(pspTotalFeat)
        pspTotalFeat = self.drop_2(pspTotalFeat)

        pspTotalFeat = self.up_2(pspTotalFeat)
        pspTotalFeat = self.drop_2(pspTotalFeat)

        pspTotalFeat = self.up_3(pspTotalFeat)
        #因为resnet的输出是其输入图像的1/8大小，所以这里执行了三次上采样操作，将特征图大小恢复到了原始尺寸

        return self.final(pspTotalFeat)






# #%%
# import torch
# import torch.nn as nn
# # m = nn.Dropout2d(p=0.2)
# # input = torch.randn(20, 16, 32, 32)
# # output = m(input)
# # print(output)
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(5)])

#     def forward(self, x):
#         # ModuleList can act as an iterable, or be indexed using ints
#         for i, l in enumerate(self.linears):
#             print(i//2,l)
#             x = self.linears[i // 2](x) + l(x)
#         return x

# s = MyModule()
# input = torch.randn(1,10)
# print(input)
# print(s(input))
# #%%
# import torch.nn as nn
# import torch
# # target output size of 5x7
# m = nn.AdaptiveAvgPool2d((5,7))
# input = torch.randn(1, 64, 8, 9)
# output = m(input)
# print(output.size(1))
# # target output size of 7x7 (square)
# m = nn.AdaptiveAvgPool2d(7)
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
# # target output size of 10x7
# m = nn.AdaptiveMaxPool2d((None, 7))
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
