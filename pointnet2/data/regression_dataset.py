import h5py
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import pdb
import os


class RegressionDataset(Dataset):
    """ Data set with distances on a mesh. """
    def __init__(self, file_path):
        """
        Args:
            file_path (str): path to h5py file
        """
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" + file_path)
        self._file = h5py.File(self.file_path, "r")
        self.data = self._file["data"]
        self.labels = self._file["labels"]
        self.index = self._file["index"]

    def __getitem__(self, i):
        data = torch.from_numpy(self.data[i, ...].astype(np.float32))
        index =self.index[i, ...].astype(np.int32)

        # TODO: Determine if we need to scale features - would that change distances?
        # point_set = data - np.expand_dims(np.mean(data, axis=0), 0)  # center
        # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        # point_set = point_set / dist  # scale

        labels = torch.from_numpy(self.labels[i, ...].astype(np.float32))
        return data, labels, index

    def __len__(self):
        num_point_clouds = self.data.shape[0]
        return num_point_clouds

