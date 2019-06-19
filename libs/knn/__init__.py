
import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable,Function
from libs.knn.knn_pytorch import knn_pytorch

class KNearestNeighbor(Function):
    """ Compute k nearest neighbors for each query point
    """
    def __init__(self,k):
        self.k = k
    
    def forward(self,ref,query):
        ref = ref.float().cuda()
        query = query.float.cuda()

        inds = torch.empty(query.shape[0], self.k, query.shape[2]).long().cuda() #Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument sizes.
        knn_pytorch.knn(ref, query, inds)

        return inds

