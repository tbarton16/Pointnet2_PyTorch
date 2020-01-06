import numpy as np
import h5py
import csv
import os
import h5py
import glob
import numpy as np
import pdb
import random
import pprint

npoints = 4096
ndim = 4
test = [13,
        19,
        23,
        38,
        39,
        51,
        60,
        66,
        68,
        75,
        78,
        80,
        88,
        90,
        93,
        96,
        106,
        108,
        114,
        116,
        126,
        142,
        152,
        153,
        163,
        190,
        197,
        199]
train = [1,
         2,
         4,
         5,
         6,
         8,
         9,
         10,
         11,
         12,
         14,
         15,
         16,
         17,
         18,
         20,
         21,
         22,
         24,
         25,
         26,
         27,
         28,
         29,
         30,
         31,
         32,
         33,
         34,
         35,
         36,
         40,
         41,
         42,
         43,
         44,
         45,
         46,
         47,
         48,
         49,
         50,
         52,
         53,
         54,
         55,
         56,
         57,
         58,
         59,
         62,
         63,
         64,
         65,
         67,
         69,
         70,
         71,
         72,
         73,
         74,
         76,
         77,
         79,
         81,
         82,
         83,
         84,
         86,
         87,
         89,
         91,
         92,
         94,
         95,
         97,
         98,
         99,
         100,
         101,
         102,
         103,
         104,
         105,
         107,
         109,
         110,
         112,
         113,
         117,
         118,
         119,
         120,
         121,
         122,
         123,
         124,
         125,
         127,
         128,
         129,
         130,
         131,
         132,
         133,
         134,
         135,
         136,
         137,
         138,
         139,
         140,
         141,
         143,
         144,
         145,
         146,
         147,
         148,
         149,
         150,
         151,
         154,
         155,
         156,
         157,
         158,
         159,
         160,
         161,
         162,
         164,
         165,
         166,
         167,
         168,
         169,
         171,
         172,
         173,
         174,
         175,
         176,
         177,
         178,
         179,
         180,
         181,
         183,
         184,
         185,
         186,
         187,
         188,
         189,
         191,
         192,
         194,
         195,
         196,
         198,
         200,
         201,
         202]


def read_point_cloud(filename, stats):
    points, distances = [], []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        dists = []
        for row in reader:
            point = [float(element) for element in row[:-1]]
            distance = float(row[-1])
            # print(distance)
            points.append(point)
            distances.append(distance)
            dists.append(distance)
        # print(filename[-7:], len(list(filter((lambda x: x == 0.), dists))) / float(len(dists)) )
        stats.append(len(list(filter((lambda x: x == 0.), dists))) / float(len(dists)))
    return np.array(points), np.array(distances), stats


def read_n_point_clouds(d):
    overall_points, overall_distances, index_names_train = [], [], []
    test_points, test_distances, index_names_test = [], [], []
    stats = []
    for ff in os.listdir(d):
        if ff != '.csv':
            # sub = os.path.join(d, ff)
            filename = os.path.join(d, ff)
            name, _ = ff.split(".")
            # r = random.random()
            file_index = int(name)

            if file_index in test:
                # for fi in os.listdir(sub):
                points, distances, stats = read_point_cloud(filename, stats)
                file_index = int(name)
                overall_points.append(points[:npoints])
                overall_distances.append(distances[:npoints])
                index_names_train.append(file_index)
            else:
                # for fi in os.listdir(sub):
                points, distances, stats = read_point_cloud(filename, stats)
                file_index = int(name)
                test_points.append(points[:npoints])
                test_distances.append(distances[:npoints])
                index_names_test.append(file_index)
    print("stats:", sum(stats) / float(len(stats)), min(stats), max(stats))
    # print([len(o) for o in overall_points])
    overall_points = np.stack(overall_points, axis=0)
    overall_distances = np.stack(overall_distances, axis=0)
    test_points = np.stack(test_points, axis=0)
    test_distances = np.stack(test_distances, axis=0)
    index_names_train = np.stack(index_names_train, axis=0)
    index_names_test = np.stack(index_names_test, axis=0)
    return overall_points, overall_distances, test_points, test_distances, index_names_train, index_names_test


def write_to_h5py(overall_points, overall_distances, index, nm):
    f = h5py.File(nm, "w")
    f["data"] = overall_points
    f["labels"] = overall_distances
    f["index"] = index


def run(version):
    d = f'/home/theresa/p/{version}'

    c = d.split("/")
    c = c[-2]+c[-1]
    point_clouds, point_cloud_distances, point_clouds_test, \
    point_distances_test, index_names, index_namest = read_n_point_clouds(d)
    write_to_h5py(point_clouds, point_cloud_distances, index_names, "/home/theresa/p/"+f"{c}train.h5")
    write_to_h5py(point_clouds_test, point_distances_test, index_namest, "/home/theresa/p/"+f"{c}test.h5")


if __name__ == "__main__":
    run(4)
