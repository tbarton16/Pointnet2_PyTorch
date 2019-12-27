from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(10)
torch.manual_seed(10)
def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]

def stack(a, num=5):
  output = []
  for i in range(num):
    output.append(a)
  return np.array(output)
def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["labels"][:]
    index = f["index"][:]
    data = data[:,:4096, :]
    # data = np.array(stack(data))
    label = label[:,:4096]
    # label = np.array(stack(label))
    return data, label, index


class Indoor3DSemSeg(data.Dataset):
    def __init__(self, num_points, file, train=True, download=True, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "indoor3d_sem_seg_hdf5_data"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = (
            "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
        )
        self.file = file
        self.train, self.num_points = train, num_points

        # all_files = _get_data_files(os.path.join(self.data_dir, "all_files.txt"))
        # room_filelist = _get_data_files(
        #     os.path.join(self.data_dir, "room_filelist.txt")
        # )
        #
        # data_batchlist, label_batchlist = [], []
        # for f in all_files:
        #     data, label = _load_data_file(os.path.join(BASE_DIR, f))
        #     data_batchlist.append(data)
        #     label_batchlist.append(label)
        #
        # data_batches = np.concatenate(data_batchlist, 0)
        # labels_batches = np.concatenate(label_batchlist, 0)
        #
        # test_area = "Area_5"
        # train_idxs, test_idxs = [], []
        # for i, room_name in enumerate(room_filelist):
        #     if test_area in room_name:
        #         test_idxs.append(i)
        #     else:overfit
        #         train_idxs.append(i)

        data, label, index = _load_data_file(self.file)
        # print(data[0][2])
        # print(label[0][2])

      # todo@tbarton and normalize the data
      #   data = data - np.expand_dims(np.mean(data, axis=0), 0)  # center
        mean = data.mean(axis=1)
        # print(data.shape)

        # print(mean.shape)
        # data = data - mean[:, np.newaxis, :]
        # print("squared", (data ** 2).shape)
        # print((np.sum(data ** 2, axis=1)).shape)
        var = (np.sqrt(np.sum(data ** 2, axis=1)))
        dist = np.maximum(var, np.zeros_like(var))
        print(dist.shape)
        # data = data / dist[:, np.newaxis, :] # scale
        if not np.all(np.isfinite(data)):
            print(data)
            assert False
        if not np.all(np.isfinite(label)):
            print(label)
            assert False
        self.points = data
        self.labels = label
        self.index = index


    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).type(
            torch.LongTensor
        )
        current_index = self.index[idx].copy()

        return current_points, current_labels, current_index

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Indoor3DSemSeg(16, "./", train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
