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

class Uvloader(data.Dataset):
    def __init__(self, points, labels, dists, ind):
        super().__init__()
        self.points = np.array(points)
        self.labels = np.array(labels)
        self.dists = np.array(dists)
        self.ind = np.array(ind)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points[0].shape[0])
        pt_idxs = np.random.choice(pt_idxs, 4000, replace=False)

        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy(

        )).type(
            torch.FloatTensor
        )

        current_dists = torch.from_numpy(self.dists[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_index = self.ind[idx].copy()

        return current_points, current_labels, current_dists, current_index


    def __len__(self):
        return int(self.points.shape[0])


class Indoor3DSemSeg(data.Dataset):
    def __init__(self, num_points, file, train=True, thresh=0, download=True, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "indoor3d_sem_seg_hdf5_data"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = (
            "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
        )
        self.file = file
        self.thresh = thresh
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
        if train:
            print("train", dist.shape)
        else:
            print("test", dist.shape)
            print(index)
            # assert False

        # data = data / dist[:, np.newaxis, :] # scale
        if not np.all(np.isfinite(data)):
            print(data)
            assert False
        if not np.all(np.isfinite(label)):
            print(label)
            assert False
        self.points = data
        if train:
            thresh = np.median(np.array(label), axis=1, keepdims= True)
            print("train median", np.median(thresh, axis=0))
            class_label = np.where(label > thresh, 1., 0.)
            print("% train above",np.average(np.sum(class_label, axis=1) / class_label.shape[1], axis=0))
        else:
            thresh = self.thresh
            print("test median", thresh)
            class_label = np.where(label > thresh, 1., 0.)
            print("% test above",np.average(np.sum(class_label, axis=1) / class_label.shape[1], axis=0))
        # print(thresh)
        # print(label.shape, )

        self.labels = class_label
        self.index = index


    def __getitem__(self, idx):
        pt_idxs = np.arange(0, 10000)
        pt_idxs = np.random.choice(pt_idxs, 4096, replace=False)

        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).type(
            torch.LongTensor
        )
        current_index = self.index[idx].copy()
        # print(current_points,)
        # print(current_labels)
        # print(current_index)
        # assert False

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
