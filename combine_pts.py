import os
import sys
import numpy as np
from numpy import genfromtxt
import random

def invert_points(pts):
    inverse_result = []
    for i in pts:
        i = i.tolist()
        if i[-1] == 0.:
            i = i[:-1]
            i.append(1.)
            inverse_result.append(np.asarray(i))
        else:
            i = i[:-1]
            i.append(0.)
            inverse_result.append(np.asarray(i))
    return np.asarray(inverse_result)

def reweight_points(pts, p):
    inverse_result = []
    for i in pts:
        i = i.tolist()
        if i[-1] == 0.:
            r = random.uniform(0,1)
            if r > p:
                inverse_result.append(np.asarray(i))
        else:
            inverse_result.append(np.asarray(i))

    return np.asarray(inverse_result)

def resample_directory(o, m, exclusion_list):
    precisions = []
    o = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/results_all_levels"
    d1 = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/fst_test/test_guessesgt"
    d2 = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/snd_00152020-01-06T08/test_guessesgt"
    d3 = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/trd_00132020-01-06T13/test_guessesgt"
    if not os.path.exists(o):
        os.mkdir(o)
    for i,f in enumerate(os.listdir(d1)[:]):
        try:
            infile1 = os.path.join(d1, f)
            num, _ = f.split(".")
            print(num)
            numpad = num
            gamma = genfromtxt(infile1, dtype=np.float64, delimiter=",")
            g2 = genfromtxt(os.path.join(d2, f), dtype=np.float64, delimiter=",")
            g3 = genfromtxt(os.path.join(d3, f), dtype=np.float64, delimiter=",")

            gamma = reweight_points(gamma, 1)
            g2 = reweight_points(g2, .5)
            result = np.vstack([gamma, g2, g3])
            print(result.shape)


            np.savetxt(os.path.join(o, f), invert_points(result), delimiter=",")

        except:
            continue



if __name__ == "__main__":
    resample_directory(1,2,3)
