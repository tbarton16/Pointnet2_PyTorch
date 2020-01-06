import sys
sys.path.append("/home/theresa/libigl/python")
import pyigl as old_igl
import igl as i
from iglhelpers import *
import numpy as np
from numpy import genfromtxt
import os
def accuracy(mesh, ind, preds):
    # first calculate diffusion for all the verts, then convert these values to probabilities
    # then sample a bunch of points and keep n_pts of them
    v, f = i.read_triangle_mesh(mesh)
    seams = f"/home/theresa/p/groundtruthseam/{ind}.csv"
    seams = genfromtxt(seams, dtype=np.float64, delimiter=",")
    gamma = genfromtxt(preds, dtype=np.float64, delimiter=",")
    gamma = np.array([list(g[:-1]) for g in gamma if int(g[-1]) == 0])
    gamma = np.array(gamma)
    _, prims, _ = i.point_mesh_squared_distance(gamma, v, f)
    gamma = []
    for p in [i for idx, i in enumerate(f) if idx in prims]:
        v1, v2, v3 = p
        gamma.append(v1)
        gamma.append(v2)
        gamma.append(v3)
    # print(gamma)
    verts = old_igl.eigen.MatrixXd()
    FI = old_igl.eigen.MatrixXi()
    tc = old_igl.eigen.MatrixXd()
    cn = old_igl.eigen.MatrixXd()
    ftc = old_igl.eigen.MatrixXi()
    fn = old_igl.eigen.MatrixXi()
    old_igl.readOBJ(mesh, verts, tc, cn, FI, ftc, fn);
    d = old_igl.heat_geodesics_data()
    old_igl.heat_geodesics_precompute(verts, FI, d)
    results = old_igl.eigen.MatrixXd()
    gamma = p2e(np.array(gamma, dtype=np.int32))
    old_igl.heat_geodesics_solve(d, gamma, results)
    _, prims, closest_vertices = i.point_mesh_squared_distance(seams, v, f)
    distances = []
    for s, tri, c in zip(seams, prims, closest_vertices):
        # verts = [v[i] for i in f[face]]
        dist_0 = [np.linalg.norm(s-i) for i in [v[j] for j in f[tri]]]

        dist_0 /= np.sum(dist_0, axis=0)
        weights = [results[j]*dist_0[i] for i, j in enumerate(f[tri])]
        total_distance = sum(weights)
        distances.append(total_distance)

    return np.average(distances)

def precision(mesh, ind, preds):
    # first calculate diffusion for all the verts, then convert these values to probabilities
    # then sample a bunch of points and keep n_pts of them
    v, f = i.read_triangle_mesh(mesh)
    gammapath = f"/home/theresa/p/groundtruthseam/{ind}.csv"
    gamma = genfromtxt(gammapath, dtype=np.float64, delimiter=",")
    preds = genfromtxt(preds, dtype=np.float64, delimiter=",")
    preds = np.array([list(g[:-1]) for g in preds if int(g[-1]) == 0])
    gamma = np.array(gamma)
    _, prims, _ = i.point_mesh_squared_distance(gamma, v, f)
    gamma = []
    for p in [i for idx, i in enumerate(f) if idx in prims]:
        v1, v2, v3 = p
        gamma.append(v1)
        gamma.append(v2)
        gamma.append(v3)
    # print(gamma)
    verts = old_igl.eigen.MatrixXd()
    FI = old_igl.eigen.MatrixXi()
    tc = old_igl.eigen.MatrixXd()
    cn = old_igl.eigen.MatrixXd()
    ftc = old_igl.eigen.MatrixXi()
    fn = old_igl.eigen.MatrixXi()
    old_igl.readOBJ(mesh, verts, tc, cn, FI, ftc, fn);
    d = old_igl.heat_geodesics_data()
    old_igl.heat_geodesics_precompute(verts, FI, d)
    results = old_igl.eigen.MatrixXd()
    gamma = p2e(np.array(gamma, dtype=np.int32))
    old_igl.heat_geodesics_solve(d, gamma, results)
    print("heatmap_min_and_max, ", min(results), max(results))
    _, prims, closest_vertices = i.point_mesh_squared_distance(preds, v, f)
    distances = []
    for p, tri, c in zip(preds, prims, closest_vertices):
        # verts = [v[i] for i in f[face]]
        dist_0 = [np.linalg.norm(p-i) for i in [v[j] for j in f[tri]]]

        dist_0/= np.sum(dist_0, axis=0)
        weights = [results[j]*dist_0[i] for i, j in enumerate(f[tri])]
        total_distance = sum(weights)
        distances.append(total_distance)

    return np.average(distances)

def sample_mesh(mesh, n_pts, gamma, output):
    print(mesh)
    # first calculate diffusion for all the verts, then convert these values to probabilities
    # then sample a bunch of points and keep n_pts of them
    v, f = i.read_triangle_mesh(mesh)
    gamma = genfromtxt(gamma, dtype=np.float64, delimiter=",")
    gamma = np.array([list(g[:-1]) for g in gamma if int(g[-1]) == 0])
    _, prims, _ = i.point_mesh_squared_distance(gamma, v, f)
    gamma = []
    for p in [i for idx,i in enumerate(f) if idx in prims]:
        v1, v2, v3 = p
        gamma.append(v1)
        gamma.append(v2)
        gamma.append(v3)
    # print(gamma)
    verts = old_igl.eigen.MatrixXd()
    FI = old_igl.eigen.MatrixXi()
    tc = old_igl.eigen.MatrixXd()
    cn = old_igl.eigen.MatrixXd()
    ftc = old_igl.eigen.MatrixXi()
    fn = old_igl.eigen.MatrixXi()
    old_igl.readOBJ(mesh, verts, tc, cn, FI, ftc, fn);
    d = old_igl.heat_geodesics_data()
    old_igl.heat_geodesics_precompute(verts, FI, d)
    results = old_igl.eigen.MatrixXd()
    gamma = p2e(np.array(gamma, dtype=np.int32))
    old_igl.heat_geodesics_solve(d, gamma, results)
    print("heatmap_min_and_max, ", min(results), max(results))
    results /= max(results)
    pts = []
    counter = 0
    while len(pts)< n_pts:
        if counter > 20:
            print("abandoning")
            return
        counter += 1
        Bary, FI = i.random_points_on_mesh(5*n_pts, v, f)
        for b, face in zip(Bary,FI):
            verts = [v[i]for i in f[face]]
            weights = [results[i] for i in f[face]]
            pt = sum([float(b[i]) * verts[i] for i in range(len(verts))])
            prob = sum([float(b[i]) * weights[i] for i in range(len(verts))])
            if prob > np.random.uniform(0,1):
                pts.append(pt)

    pts = pts[:n_pts]
    pts = np.array(pts)
    np.savetxt(output, pts, delimiter=",")


def resample_directory(d, o, m, exclusion_list):
    if not os.path.exists(o):
        os.mkdir(o)
    for i,f in enumerate(os.listdir(d)[:]):
        infile = os.path.join(d, f)
        num, _ = f.split(".")
        numpad = num
        if int(num) in exclusion_list:
            continue
        if len(num) == 1:
            num = "00" + num
        elif len(num) == 2:
            num = "0" + num
        outfile = os.path.join(o,num+".csv")
        mfile = os.path.join(m, num + ".obj")

        try:

            sample_mesh(mfile, 4096, infile, outfile)
            print("precision: ", precision(mfile, numpad, infile))
            print("accuracy: ", accuracy(mfile, numpad, infile))
        except:
            pass


if __name__ == "__main__":
    from datetime import datetime

    # Add code to modify contents of geo.
    # Use drop down menu to select examples.
    tm = datetime.now().isoformat()
    tm = tm[:-13]
    # print(precision("/home/theresa/p/datav5_obj/081.obj", 81,
    #                 "/home/theresa/Pointnet2_PyTorch/generated_shapes_debug/train_guessesgt/81.csv"))
    print("TRAIN")
    resample_directory("/home/theresa/Pointnet2_PyTorch/generated_shapes_debug/train_guesses",
                  "/home/theresa/Pointnet2_PyTorch/reconstructed_seams/train_output",
                  "/home/theresa/p/datav5_obj", [5,111,112, 42, 83,128])
    print("TEST")
    resample_directory("/home/theresa/Pointnet2_PyTorch/generated_shapes_debug/test_guesses",
                  "/home/theresa/Pointnet2_PyTorch/reconstructed_seams/test_output",
                  "/home/theresa/p/datav5_obj", [])
