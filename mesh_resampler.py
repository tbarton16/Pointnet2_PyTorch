import sys
sys.path.append("/home/theresa/libigl/python")
import pyigl as old_igl
import igl as ig
from iglhelpers import *
import numpy as np
from numpy import genfromtxt
import os

#  use eval results to generate next samples. samples are saved to reconstructed seams with file name, v.
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

class facenn:
    def __init__(self, face, verts):
        self.face = face
        self.verts = verts
        self.neighbors = []


def facetoface(f):
    def get_edges(tri):
        edges = []
        for t in tri:
            for v in tri:
                if t > v:
                    edge = (t,v)
                else:
                    edge = (v,t)
                if edge not in edges and t != v:
                    edges.append(edge)
        return edges
    # verts = [[3,2,6], [3,1,2], [1,4,2],[3,2,6], [3,1,2], [1,4,2]]
    # f = [[3,2,6], [3,1,2], [1,4,2], [1,5,4]]
    edgetoface = {}
    idxtofacenn ={}

    face = facenn(0, f[0])
    idxtofacenn[0] = face
    a, b, c = get_edges(f[0])
    edgetoface[a] = [0]
    edgetoface[b] = [0]
    edgetoface[c] = [0]

    for i, fi in enumerate(f[1:],start=1):
        idxtofacenn[i] = facenn(i, fi)
        for edge in  get_edges(fi):
            if edge in edgetoface:
                neighbor = idxtofacenn[edgetoface[edge][0]]
                idxtofacenn[i].neighbors.append(neighbor)
                neighbor.neighbors.append(idxtofacenn[i])
            else:
                edgetoface[edge] = [i]

    # print([i.face for i in idxtofacenn[0].neighbors])
    # print([i.face for i in idxtofacenn[1].neighbors])
    # print([i.face for i in idxtofacenn[2].neighbors])
    print([i.face for i in idxtofacenn[3].neighbors])
    return idxtofacenn

def facetopoints(points, v, f):
    gamma = genfromtxt(points, dtype=np.float64, delimiter=",")
    # gamma = np.asarray([[0.,0.1,0.,0], [0.,0.5,0.,1]], dtype=np.float64)
    pointsonly = np.array([list(g[:-1]) for g in gamma])
    #print(pointsonly)
    # ig.write_triangle_mesh("testmesh.obj", v, f)
    _, prims, _ = ig.point_mesh_squared_distance(pointsonly, v, f)
    ftop = {}
    # print("idx, face", prims[0], f[prims[0]])
    for fi, z in zip(prims.tolist(), gamma):
        if fi in ftop:
            ftop[fi].append(z)
        else:
            ftop[fi] = [z]
    return ftop

def verttoface(f):
    # Map from vertex idx to faces it is a part of
    v2f = {}
    for i, fa in enumerate(f):
        for vert in fa:
            if vert in v2f:
                v2f[vert].append(i)
            else:
                v2f[vert] = [i]
    return v2f


def vert_knn(mesh, n_pts, gamma, output, k = 5):
    v, f = ig.read_triangle_mesh(mesh)
    # v = np.asarray([[0, 2, 0], [1, 1, 0], [-1, 1, 0], [1, 2, 0], [-1, 3, 0], [0, 0, 0]], dtype=np.float64)
    # f = np.asarray([[3, 2, 6], [3, 1, 2], [1, 4, 2], [1, 5, 4]], dtype=np.int32)
    # f = np.asarray([[i-1 for i in p] for p in f], dtype=np.int32)
    # # print(v, f)
    ftop = facetopoints(gamma, v, f)
    idxtofacenn = facetoface(f)
    v2f = verttoface(f)
    labeledverts = [] # mapping from vertex index to faces
    for i, verts in enumerate(v):
        fringe = v2f[i]
        closepoints = []
        # print("fringe", fringe)
        while len(closepoints) < k:
            points =[]
            for p in fringe:
                if p not in ftop:
                    continue # no points on prim
                for pt in ftop[p]:
                    points.append(pt)

            if len(points) < k-len(closepoints):
                closepoints += points
            else:
                points = sorted(points, key=lambda x: np.linalg.norm(np.asarray(x[:-1])- np.asarray(verts)))
                closepoints += points[:k-len(closepoints)]
            newfringe = []

            for p in fringe:
                for n in idxtofacenn[p].neighbors:
                    newfringe.append(n.face)
            fringe = newfringe
        tot = sum([np.linalg.norm(np.asarray(x[:-1])- np.asarray(verts)) for x in closepoints])
        weight = 0.
        # print("closest", closepoints)
        for pt in closepoints:
            d = np.linalg.norm(np.asarray(pt[:-1])- np.asarray(verts))
            # print("d",float(tot))
            weight += pt[3] * (1.-(d/float(tot)))
        if weight < .8:
            weight = 0.
        verts = verts.tolist()
        verts.append(weight)
        labeledverts.append(np.asarray(verts))
    # print(labeledverts)
    # import time
    # time.sleep(10)

    np.savetxt(output, np.asarray(labeledverts), delimiter=",")










def resample_directory(d, o, m, exclusion_list):
    precisions = []
    accs = []
    if not os.path.exists(o):
        os.mkdir(o)
    inclusion_list = [136, 48, 138, 70, 80, 55, 75, 192]
    for i,f in enumerate(os.listdir(d)[:]):
        infile = os.path.join(d, f)
        num, _ = f.split(".")
        print(num)
        numpad = num
        if int(num) not in inclusion_list: # TODO change back
            continue
        if len(num) == 1:
            num = "00" + num
        elif len(num) == 2:
            num = "0" + num
        outfile = os.path.join(o,num+".csv")
        mfile = os.path.join(m, num + ".obj")

        # try:
            # p =  precision(mfile, numpad, infile)
            # precisions.append(p)
            # a = accuracy(mfile, numpad, infile)
            # accs.append(a)
        vert_knn(mfile, 4096, infile, outfile)
            # print("precision: ",p)
            # print("accuracy: ", a)
        # except:
        #     pass
    return precisions, accs


if __name__ == "__main__":

    v = str(sys.argv[1])
    train_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/{v}/train_guesses"
    test_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/{v}/test_guesses"
    meshes = "/home/theresa/p/datav5_obj"
    train_output = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/{v}+resampled/"
    test_output = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/{v}+resampled/"
    if not os.path.isdir(train_output):
        os.makedirs(train_output)
    # train_output += "train_resampled"
    # test_output += "test_resampled"
    if not os.path.isdir(test_output):
        os.makedirs(test_output)
   # if not os.path.isdir(test_output):
   #    os.makedirs(test_output)

    #
    print("TEST")
    testp, testa = resample_directory(test_predicted_path,
                                      train_output,
                                      meshes, [68, 128, 125])
    print("TRAIN")
    trainp, traina = resample_directory(train_predicted_path, train_output,
                  meshes, [0, 5, 83, 112, 173, 191])

    print("train:", np.average(trainp), np.average(traina))
    print("test:", np.average(testp), np.average(testa))
