
import torch.utils.data as data
import yaml
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import numpy.ma as ma
import torch
import random


class PoseDataset(data.Dataset):
    def __init__(self, mode, pooledImgSize, add_noise, root, noise_trans, refine):
        self.objlist = [1,2,4,5,6,8,9,10,11,12,13,14,15]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}  
        self.pt = {}

        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.numOfChoosedPoints = pooledImgSize * pooledImgSize
        self.pooledImgSize = pooledImgSize
        self.add_noise = add_noise

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count %10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                self.list_obj.append(item)
                self.list_rank.append(int(input_line))
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' %item),'r')
            self.meta[item] = yaml.load(meta_file)
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' %item))

            print("Object {0} buffer loaded".format(item))
        
        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) # [第一行 0000000，第二行11111111，……]
        self.ymap = np.array([[i for i in range(640)] for j in range(480)]) # 【第一行 012345，第二行012345，……】

        self.transcolor = transforms.ColorJitter(0.2,0.2,0.2,0.05)
        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rank = self.list_rank[index]

        if obj == 2:
            for i in range(0,len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth =  ma.getmaskarray(ma.masked_not_equal(depth, 0))
        
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label,np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label,np.array([255,255,255])))[:,:,0]
        
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.transcolor(img)
        
        img = np.array(img)[:,:,:3]
        img = np.transpose(img, (2, 0, 1)) # 交换img的通道，H×W×3 --》 3×H×W
        img_masked = img

        rmin,rmax,cmin,cmax = get_bbox(meta['obj_bb'])

        img_masked = img_masked[:,rmin:rmax,cmin:cmax]

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3,3))
        target_t = np.array(meta['cam_t_m2c'])
        

        choose = mask[rmin:rmax,cmin:cmax].flatten().nonzero()[0]#是将bbox中的非0的像素索引（一维）保存
        # print('>>>>>>>>----------pixels not equal to 0 :  {0} ---------<<<<<<<<'.format(len(choose)))
        
        # 新加代码，实现将不规则物体规则化
        # #****************************----start-----******************************************
        # 选点
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc,cc,cc,cc,cc,cc)
        if len(choose) > self.numOfChoosedPoints:
            c_mask = np.zeros(len(choose),dtype=int)
            c_mask[:self.numOfChoosedPoints] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose,(0,self.numOfChoosedPoints - len(choose)), 'wrap') 
        
        #计算选择的点的点云坐标        
        depth_masked = depth[rmin:rmax,cmin:cmax].flatten()[choose][:,np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax,cmin:cmax].flatten()[choose][:,np.newaxis].astype(np.float32)#mask区域的点的行数
        ymap_masked = self.ymap[rmin:rmax,cmin:cmax].flatten()[choose][:,np.newaxis].astype(np.float32)#mask区域的点的列数
        cam_scale = 1.0
        # 将深度图转化为点云，计算依据是坐标的变换关系（点云坐标是相机坐标系下的坐标）
        pt2 = depth_masked / cam_scale # 点云的z
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx #点云的x
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy #点云的y
        cloud = np.concatenate((pt0, pt1, pt2),axis = 1) #cloud self.numOfChoosedPoints行，3列
        cloud = np.add(cloud, -1.0*target_t) / 1000.0
        cloud = np.add(cloud, target_t/1000.0) #将单位由毫米换成米

        add_t = np.array([random.uniform(-self.noise_trans,self.noise_trans) for i in range(3)])
        if self.add_noise:
            cloud = np.add(cloud,add_t)

        img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
        img_masked = img_masked.numpy()
        img_c0 = img_masked[0,:,:].flatten()[choose][:,np.newaxis].astype(np.float32)
        img_c1 = img_masked[1,:,:].flatten()[choose][:,np.newaxis].astype(np.float32)
        img_c2 = img_masked[2,:,:].flatten()[choose][:,np.newaxis].astype(np.float32)
        img_choose = np.concatenate((img_c0, img_c1, img_c2),axis = 1) #self.numOfChoosedPoints行，3列
        self.norm(torch.from_numpy(img_masked.astype(np.float32)))

        img_cloud = np.concatenate((img_choose, cloud),axis = 1) #self.numOfChoosedPoints行，6列
        img_cloud = np.transpose(img_cloud, (1, 0))
        img_cloud = img_cloud.reshape(6,self.pooledImgSize,self.pooledImgSize)

        #****************************----end-----******************************************
        
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0) # 删掉一些点，保留num_pt_mesh_small个点

        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0
        
        return torch.from_numpy(img_cloud.astype(np.float32)),\
               torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)]),\
               ori_img
        
        # return self.norm(torch.from_numpy(img_masked.astype(np.float32))),\
        #        torch.from_numpy(cloud.astype(np.float32)), \
        #        torch.LongTensor(choose.astype(np.int32)), \
        #        torch.from_numpy(target.astype(np.float32)), \
        #        torch.from_numpy(model_points.astype(np.float32)), \
        #        torch.LongTensor([self.objlist.index(obj)]),\
        #        ori_img
                   # torch.LongTensor([self.objlist.index(obj)])返回obj位于objlist的哪个位置

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox): #box 可能的大小都是一致的
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

# #%%
# import numpy as np
# import random
# # 1
# dellist = [j for j in range(0, 10)]
# print(dellist)
# dellist = random.sample(dellist, 8)
# print(dellist)

# # 2
# # xmap = np.array([[j for i in range(10)] for j in range(8)])
# # ymap = np.array([[i for i in range(10)] for j in range(8)])
# # # print(xmap)
# # # print(ymap)
# # xmap_masked = xmap[2:5,2:5].flatten()[:,np.newaxis].astype(np.float32)
# # ymap_masked = ymap[2:5,2:5].flatten()[:,np.newaxis].astype(np.float32)
# # # print(xmap_masked)
# # # print(ymap_masked)
# # cloud = np.concatenate((xmap_masked, ymap_masked),axis = 1)
# # temp = np.array([1,2])
# # cloud = np.add(cloud,-1.0* temp)/100.0;
# # cloud = np.add(cloud, temp/100.0)
# # print(cloud)
# #%%
# import os
# import os.path
# import yaml
# import numpy as np
# # print(os.path.dirname)
# input_file = open('/home/galen/deepLearning/poseEstimation/PatchN/datasets/test.txt')
# while 1:
#     input_line = input_file.readline()
#     print(input_line[0:])
#     if not input_line:
#         print('over')
#         break
# # meta = []
# # meta_file = open('/home/galen/deepLearning/poseEstimation/Patch/datasets/gt.yml','r')
# # meta = yaml.load(meta_file)
# # print(meta[1][0])


# #%%
# import numpy.ma as ma
# import numpy as np
# depth = np.array([[[1,2,2],[4,5,5],[4,5,5]],[[7,8,5],[9,10,11],[4,5,5]]])
# print(depth)
# print(depth.shape)

# mask_depth =  ma.getmaskarray(ma.masked_equal(depth, np.array([4,5,5])))[:,:,0]
# print(mask_depth)


# #%%
# import torch.utils.data as data
# import os
# import os.path
# import yaml
# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy.ma as ma

# test = Image.open('/home/galen/deepLearningCode/PoseEstimation/Patch/datasets/color.png')
# print(np.array(test).shape)
# mask_depth =  ma.getmaskarray(ma.masked_equal(test, np.array([1,7,11])))
# img = np.array(test)[:, :, :3]
# print(img.shape)

# choose = img.flatten().nonzero()[0]
# print(type(choose))
# choose = np.array([choose])
# print(type(choose))

# # for i in range(len(choose)):
# #     print(choose[i])
# img = np.transpose(img,(2,0,1))
# print(img.shape)

# # print(mask_depth)