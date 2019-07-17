import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# def conv3x3(in_channels,out_channels,stride=1,dilation = 1):
#     return nn.Conv2d(in_channels,out_channels,kernel_size=3 ,stride = stride,
#                     padding=1,dilation=dilation,bias=False)
def conv3x3(in_channels,out_channels,stride=1,dilation = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3 ,stride = stride,
                    padding=dilation,dilation=dilation,bias=False)

class basicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,dilation = 1, downsample=None):
        super(basicBlock,self).__init__()
        self.stride = stride
        self.layer1 = conv3x3(in_channels,out_channels,stride=stride,dilation=dilation)
        self.layer2 = conv3x3(out_channels,out_channels, stride=1, dilation=dilation)#layer2都不downsample和改变channel数量
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,input):     
        output = self.layer1(input)
        output = self.relu(output)

        output = self.layer2(output)

        # if self.stride == 1:
        #     output = input + output
        # else:
        #     output = output + F.conv2d(input,input.size(1),kernel_size=1,stride=self.stride,bias=False)
        if self.downsample is not None:
            residual = self.downsample(input)
        else:
            residual = input
        
        output += residual

        output = self.relu(output)  
        return output

class ResNet(nn.Module):#整个resnet，将图像大小变为了原来的1/8
    # out_channelNum = [64,128,256,512]
    # in_channelNum = [64,64,128,256]
    def __init__(self,block,sizes=(2,2,2,2)):
        self.inplanes = 64
        super(ResNet,self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            # nn.Conv2d(6,64,kernel_size=3,stride=1,padding =1 ,bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ) # pre这一些层，将图像缩小了4倍（Conv2d和MaxPool2d）
        # self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        # self.batchNorm = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.stages = []
        # self.stages = nn.ModuleList([self._make_stage(sizes[i],in_channelNum[i],out_channelNum[i]) for i in len(sizes)])
        self.avgPool = nn.AvgPool2d(kernel_size=2,stride = 2)

        self.layer1 = self._make_layer(block,  64, sizes[0])
        self.layer2 = self._make_layer(block, 128, sizes[1], stride=2) #将图像大小减半
        self.layer3 = self._make_layer(block, 256, sizes[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, sizes[3], stride=1, dilation=4)
        #layer3,4stride为1，所以并不像原始的resnet一样会把特征图尺寸减半，而又为了保持较大的感受野，所以用了dilation
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()                

    def _make_layer(self,block,out_channelNum,num_block,stride=1, dilation=1):
        downsample = None  #downsample 不是仅仅下采样，也有channel变换。在本代码中，只有stride=2的时候，执行下采样。
        #在layer3,4中，只执行通道数变换
        if stride != 1 or self.inplanes != out_channelNum * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,out_channelNum * block.expansion,
                            kernel_size=1, stride=stride,bias=False)
            )
        layers = [block(self.inplanes,out_channelNum,stride,downsample=downsample)]
        self.inplanes = out_channelNum*block.expansion
        for i in range(1,num_block):
            layers.append(block(self.inplanes,out_channelNum,dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self,img):
        feature = self.pre(img)
        # feature = self.relu(feature)
        # feature = self.maxPool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        class_s = self.layer3(feature)
        feature = self.layer4(class_s)
        return feature,class_s


def resnet18(pretrained=False):
    model = ResNet(basicBlock,[2,2,2,2])
    return model

def resnet34(pretrained=False):
    model = ResNet(basicBlock, [3, 4, 6, 3])
    return model

# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             # nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 # nn.BatchNorm2d(outchannel)
#             )
#     def forward(self, x):
#         out = self.left(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, ResidualBlock, num_classes=10):
#         super(ResNet, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             # nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out_c = self.layer3(out)
#         out = self.layer4(out_c)
#         # out = F.avg_pool2d(out, 4)
#         # out = out.view(out.size(0), -1)
#         # out = self.fc(out)
#         return out,out_c


# def resnet18(pretrained=False):
#     return ResNet(ResidualBlock)

