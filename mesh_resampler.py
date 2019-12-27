import sys

sys.path.append("/home/theresa/libigl/python")

import pyigl as old_igl
import igl as i
from iglhelpers import *
import numpy as np
from numpy import genfromtxt
import csv

def sample_mesh(mesh, n_pts, gamma):
    # first calculate diffusion for all the verts, then convert these values to probabilities
    # then sample a bunch of points and keep n_pts of them
    output = "/home/theresa/p/diffusedpoints3.csv"
    # gamma = "/home/theresa/p/points3.csv"
    # infile = "/home/theresa/p/out3.obj"
    verts = old_igl.eigen.MatrixXd()
    faces = old_igl.eigen.MatrixXi()
    tc = old_igl.eigen.MatrixXd()
    cn = old_igl.eigen.MatrixXd()
    ftc = old_igl.eigen.MatrixXi()
    fn = old_igl.eigen.MatrixXi()
    old_igl.readOBJ(mesh, verts, tc, cn, faces, ftc, fn);

    d = old_igl.heat_geodesics_data()
    old_igl.heat_geodesics_precompute(verts, faces, d)
    results = old_igl.eigen.MatrixXd()

    gamma = genfromtxt(gamma, dtype=np.int32, delimiter='\n')
    gamma = p2e(gamma)

    old_igl.heat_geodesics_solve(d, gamma, results)
    v, f = i.read_triangle_mesh(mesh)
    Bary, faces = i.random_points_on_mesh(n_pts, v, f)
    # with open(output, "w+") as f:
    #     for r in results:
    #         f.write(str(r) + "\n")
    pts = []
    probs = []
    for b, face in zip(Bary,f):
        verts = [v[i]for i in face]
        weights = [results[i] for i in face]
        pt = sum([float(b[i]) * verts[i] for i in range(len(verts))])
        prob = sum([float(b[i]) * weights[i] for i in range(len(verts))])
        pts.append(pt)
        probs.append(prob)
        # print(b,f,verts,pt)
    print(min(probs ),max(probs))
    pts = np.array(pts)
    # np.savetxt(output, pts, delimiter=",")



if __name__ == "__main__":
    sample_mesh("/home/theresa/p/out3.obj", 1000, "/home/theresa/p/points3.csv")
