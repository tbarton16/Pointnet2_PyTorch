import numpy as np
import h5py
import csv
import os
import h5py
import glob
import numpy as np
import pdb
import random
npoints = 4096
ndim = 4

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
        stats.append( len(list(filter((lambda x: x == 0.), dists))) / float(len(dists)))
    return np.array(points), np.array(distances), stats
def read_n_point_clouds():
    overall_points, overall_distances,index_names_train = [], [], []
    test_points, test_distances, index_names_test  = [], [],[]
    d ='/home/theresa/p/datav1_balanced'
    stats =[]
    for ff in os.listdir(d):
        if ff != '.csv':
            filename = os.path.join(d, ff)
            points, distances, stats = read_point_cloud(filename, stats)
            r = random.random()
            file_index, _  = ff.split('.')
            file_index = int(file_index)
            if r < .85:
                overall_points.append(points[:npoints])
                overall_distances.append(distances[:npoints])
                index_names_train.append(file_index)
            else:
                test_points.append(points[:npoints])
                test_distances.append(distances[:npoints])
                index_names_test.append(file_index)
    print("stats:", sum(stats)/float(len(stats)), min(stats), max(stats))
    # print([len(o) for o in overall_points])
    overall_points = np.stack(overall_points, axis=0)
    overall_distances = np.stack(overall_distances, axis=0)
    test_points = np.stack(test_points, axis=0)
    test_distances = np.stack(test_distances, axis=0)
    index_names_train= np.stack(index_names_train, axis=0)
    index_names_test = np.stack(index_names_test , axis=0)
    return overall_points, overall_distances, test_points, test_distances,index_names_train,index_names_test
def write_to_h5py(overall_points, overall_distances,index, nm):

    f = h5py.File(nm,"w")
    f["data"] = overall_points
    f["labels"] = overall_distances
    f["index"] = index
if __name__ == '__main__':
    point_clouds, point_cloud_distances , point_clouds_test, point_distances_test, index_names, index_namest = read_n_point_clouds()
    write_to_h5py(point_clouds, point_cloud_distances,index_names, "../datav1balancedtrain.h5")
    write_to_h5py(point_clouds_test, point_distances_test, index_namest, "../datav1balancedtest.h5")
