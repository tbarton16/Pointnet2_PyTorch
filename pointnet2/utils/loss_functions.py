import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
import math

def robust_norm(var, dim=2):
    return ((var ** 2).sum(dim=dim) + 1e-8).sqrt()

class PRLoss():
    def __init__(self, device):
        self.dimension = 3
        self.k = 1
        #
        self.device = device
        self.gpu_id = torch.cuda.current_device()
        self.faiss_gpu = hasattr(faiss, 'StandardGpuResources')

        if self.faiss_gpu:
            # we need only a StandardGpuResources per GPU
            self.res = faiss.StandardGpuResources()
            # self.res.setTempMemoryFraction(0.1)
            self.res.setTempMemory(
                4 * (1024 * 1024 * 1024)
            )  # Bytes, the single digit is basically GB)
            self.flat_config = faiss.GpuIndexFlatConfig()
            self.flat_config.device = self.gpu_id

        # place holder
        self.forward_loss = torch.FloatTensor([0])
        self.backward_loss = torch.FloatTensor([0])

    def build_nn_index(self, database):
        """
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        """
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)

        if self.faiss_gpu:
            index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, index_cpu)
        else:
            index = index_cpu

        index.add(database)
        return index

    def search_nn(self, index, query, k):
        """
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        """
        D, I = index.search(query, k)

        D_var = torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        if self.gpu_id >= 0:
            D_var = D_var.to(self.device)
            I_var = I_var.to(self.device)

        return D_var, I_var

    def getAvgDist(self, index, query):
        D, I = index.search(query, 2)

        m_d = math.sqrt(np.percentile(D[:,1],90))
        return m_d

    def getOpMatch(self, points):
        return (points.max(axis = 0) - points.min(axis = 0)).max() / 100

    def score(self, preds, labels):
        """
        :param predict: BxN Variable in GPU
        :param labels: BxN Variable in GPU
        :return:
        """
        if self.gpu_id >= 0:
            preds = preds.to(self.device)
            labels = labels.to(self.device)

        preds = preds.size()
        labels = labels.size()


        ones = np.ones(preds.shape)

        dt = dist

        precision = (100 / ones.shape[0]) * np.sum(ones[preds < dt])
        recall = (100 / ones.shape[0]) * np.sum(ones[preds < dt])

        return (2*precision*recall) / (precision + recall + 1e-8)