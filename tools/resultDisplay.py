#%%
# from PIL import Image,ImageFilter,ImageEnhance
# #Read
# imageim = Image.open('test.png')
# #Display 
# imageim.show()
# enh = ImageEnhance.Contrast(imageim)
# enh.enhance(1.8).show("30% more contrast")

# #%%
# import matplotlib.pyplot as plt #matplotlib inline
# from skimage import data,filters
# image = data.coins()# ... or any other NumPy array!
# edges = filters.sobel(image)
# plt.imshow(edges, cmap='gray')


#%%
import cv2
import numpy as np
import torch
A = cv2.imread("/home/galen/deepLearningCode/PoseEstimation/PatchP/test.png")
ori_img = np.array(A)
ori_img=torch.from_numpy(ori_img.astype(np.float32))
print(ori_img.type())
ori_img = np.array(ori_img)
print(ori_img.shape)
cv2.imshow('test',ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
