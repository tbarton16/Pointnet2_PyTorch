import sys
sys.path.append("/home/theresa/libigl/python")
# import pyigl as old_igl
import igl as ig
import multiprocessing as mp
# from iglhelpers import *
import numpy as np
from numpy import genfromtxt
import os
import networkx as nx

is_training_output = True
test_run = False
if is_training_output:
    has_normal = -5
    pred_probability = 6
    gt_seam_dist = 7
else:
    semwi = .036
    has_normal = -4
    pred_probability = 6
    gt_seam_dist = 6

Verbose = False
include_prob = False
ct = 4096
k=3
pos_only = True
try_catch = True

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
    while len(pts) < n_pts:
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
    return idxtofacenn

def facetopoints(points, v, f, sampled_points=None):
    if points:
        gamma = genfromtxt(points, dtype=np.float64, delimiter=",")
        # if test_run:
        #     gamma = np.asarray([[0.,0.1,0.,0], [0.,0.5,0.,1],  [0., 0.6, 0., 1]], dtype=np.float64)
        pointsonly = np.array([list(g[:-4]) for g in gamma])

    else:
        gamma = sampled_points
        pointsonly = np.array([list(g[:3]) for g in gamma])
    # ig.write_triangle_mesh("testmesh.obj", v, f)
    _, prims, _ = ig.point_mesh_squared_distance(pointsonly, v, f)
    ftop = {}
    pointstoface = {}
    # print("idx, face", prims[0], f[prims[0]])
    for idx, fiz in enumerate(zip(prims.tolist(), gamma)):
        fi, z = fiz
        ftop[fi] =  ftop.get(fi, []) + [z]
        pointstoface[idx] = [fi]
    return ftop, pointstoface

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


def mesh_knn(mesh, n_pts, gamma, output, k):
    # for mesh with predictions gamma if point type is vertex, calculate the knn for each vertex
    # if point type is random, use gamma to calculate knn for passed in points
    v, f = ig.read_triangle_mesh(mesh)
    if test_run:
        v = np.asarray([[0, 2, 0], [1, 1, 0], [-1, 1, 0], [1, 2, 0], [-1, 3, 0], [0, 0, 0]], dtype=np.float64)
        f = np.asarray([[3, 2, 6], [3, 1, 2], [1, 4, 2], [1, 5, 4]], dtype=np.int32)
        f = np.asarray([[i-1 for i in p] for p in f], dtype=np.int32)

    # map faces to points
    print(mesh)
    ftop, _ = facetopoints(gamma, v, f)
    idxtofacenn = facetoface(f)
    # if point_type == "vertex":
    #     # mapping from vertex index to faces
    #     v2f = verttoface(f)
    #     poi = v
    # elif point_type == "random":
    pts = []
    while len(pts) < n_pts:
        Bary, FI = ig.random_points_on_mesh(n_pts, v, f)
        ws_points = []
        point_faces = []
        pointstoface = {}
        for idx, bf in enumerate(zip(Bary, FI)):
            b, face = bf
            verts = [v[i] for i in f[face]]
            pt = sum([float(b[i]) * verts[i] for i in range(len(verts))])
            if pos_only and pt[0] < 0.:
                continue
            ws_points.append(pt)
            point_faces.append(face)

        if not pos_only:
            opposite_points = np.array([[-1 * x, y, z] for x, y, z in ws_points])
            _, opp_prims, _ = ig.point_mesh_squared_distance(opposite_points, v, f)
            opposite_point_faces = opp_prims.astype(np.int64)

        for idx, _ in enumerate(ws_points):
            face = point_faces[idx]
            if not pos_only:
                opposite_point_face = opposite_point_faces[idx]
                pointstoface[idx] = [face, opposite_point_face]
            else:
                pointstoface[idx] = [face]
        if test_run:
            ws_points = np.asarray([[0., 0.1, 0.], [0., 1.5, 0.]], dtype=np.float64)
        vprobs, labeled_verts = vert_knn(ftop, idxtofacenn, pointstoface,
                                      ws_points, k=k)
        # _, pointstoface = facetopoints(None, v, f, sampled_points)
        # gamma = np.asarray([[0., 0.1, 0.], [0., 0.5, 0.]], dtype=np.float64)
        for p,vert in zip(vprobs, labeled_verts):
            if p < np.random.uniform(0,1):
                pts.append(vert)
            # pts.append(vert[[face]  # :])
        # print(f"points sampled: {len(pts) / float(len(labeled_verts))}" )
    pts = pts[:n_pts]
    print(np.median(np.array([p[-1] for p in pts])))
    np.savetxt(output, np.asarray(pts), delimiter=",")

import time
def vert_knn(ftop, idxtofacenn, v2f, sampled_points, k=5,seam_threshold=0,
             distance_threshold=.5):

    labeled_verts = []
    probs = []
    starttime=time.gmtime()
    # print("starttime:", time.strftime("%Y-%m-%d %H:%M:%S", starttime))
    oob = 0
    for i, verts in enumerate(sampled_points):
        # Check my prim first
        ct = False
        fringe = v2f[i]
        close_points = []
        visited_prims = [f for f in fringe]
        while len(close_points) < k and len(fringe) != 0:
            points = []
            for p in fringe:
                # no points on prim
                if p not in ftop:
                    continue
                for pt in ftop[p]:
                    points.append(pt)

            if len(points) < k - len(close_points):
                close_points += points
            else:
                # add all the points on the fringe to the points in sorted order
                # points = sorted(points, key=lambda x: np.linalg.norm(np.asarray(x[:-1])- np.asarray(verts)))
                points = sorted(points, key=lambda x: np.linalg.norm(
                    np.asarray(x[:has_normal])- np.asarray(verts)))
                close_points += points[:k-len(close_points)]
            new_fringe = []

            for p in fringe:
               for n in idxtofacenn[p].neighbors:
                    if n.face not in visited_prims:
                        new_fringe.append(n.face)
                        visited_prims.append(n.face)
            fringe = new_fringe
        tot = sum([np.linalg.norm(np.asarray(x[:has_normal]) - np.asarray(
            verts)) for x in close_points])
        weights = []
        if len(close_points) == 0:
            if Verbose:
                print("no close points for ", verts)
            continue

        # if there is only one close point, use it as the nearest neighbor
        if len(close_points) == 1:
            prob = close_points[0][pred_probability]
        else:
            close_points = sorted(close_points, key=lambda x: np.linalg.norm(
                np.asarray(x[:has_normal]) - np.asarray(verts)))
            distances = []
            for pt in close_points: # TODO weight special for sparse pts or max radius
                d = np.linalg.norm(np.asarray(pt[:has_normal])- np.asarray(
                    verts))
                distances.append(d)
                weights.append(1. - (d / float(tot)))
            if min(distances) > distance_threshold:
                prob = 1.
                if Verbose:
                    print("too far away", verts)
            else:
                # print("total weight", sum(weights))
                # print("-----")
                weights /= sum(weights)
                prob = 0.
                for w, pt in zip(weights, close_points): # TODO weight special for sparse pts or max radius
                    if not is_training_output:
                        x = min(pt[pred_probability], semwi)
                        x = 1. -  (x / semwi)
                    else:
                        x = pt[pred_probability]
                    if not ( x <= 1.0 and x >= 0.):
                        # print(f'{x} out of bounds for {pt}')
                        if not ct:
                            oob += 1
                            ct = True
                    else:
                        # print(w, x)
                        prob += w * x #pt[pred_probability]


        # vert prob
        if is_training_output:
            probs.append(prob)
        else:
            probs.append(1-prob)
        # copy over normal and distance from closest points
        normal = (close_points[0][3:6]).tolist()
        gt_dist = close_points[0][gt_seam_dist]
        verts = verts.tolist()
        verts += normal
        verts.append(gt_dist)
        labeled_verts.append(np.asarray(verts))
    # print("time to find nn:", time.strftime("%Y-%m-%d %H:%M:%S",
    #                                         time.gmtime()))
    if oob > 0:
        print(f"{oob} pts out of bounds")
    if test_run:
        print("labeled_verts", labeled_verts)

    return probs, labeled_verts


def run_extract_seams(mesh, gt, ind=310):
    seam_file = f"/home/theresa/p/vert_pred/{ind}.txt"
    # seams = f"/home/theresa/p/groundtruthseam/{ind}.csv"
    extract_seams(pts=gt, mesh=mesh, outfile=seam_file)


def extract_seams(pts, mesh, outfile):

    if test_run:
        v = np.asarray([[0, 2, 0], [1, 1, 0], [-1, 1, 0], [1, 2, 0], [-1, 3, 0], [0, 0, 0],
                        [1,4,0], [2,3,0]], dtype=np.float64)
        f = np.asarray([[3, 2, 6], [3, 1, 2], [1, 4, 2], [1, 5, 4],
                        [1, 3, 5], [7,5,4],[8,7,4],[8,4,2]], dtype=np.int32)
        f = np.asarray([[i - 1 for i in p] for p in f], dtype=np.int32)
    else:
        gamma = genfromtxt(pts, dtype=np.float64, delimiter=",")
        v, f = ig.read_triangle_mesh(mesh)

    if test_run:
        # gamma = np.asarray([[0., 0.1, 0., 1], [0., 0.5, 0., .9], [0., 0.6, 0., .5],
        #                     [0, 1.9, 0, .01], [0.5, 1.9, 0, .1], [-1, 3, 0, 1],
        #                     ], dtype=np.float64)
        gamma = np.asarray([[0, 2, 0], [1, 1, 0], [-1, 1, 0], [1, 2, 0], [-1, 3, 0], [0, 0, 0],
                        [1, 4, 0], [2, 3, 0]], dtype=np.float64)
        labels = [1,0,0,1,0,0,0,0]
        gamma = gamma.tolist()
        gamma = zip(gamma, labels)
        gamma = [g + [l] for g, l in gamma]
        print(gamma)

    v2f = verttoface(f)
    idxtofacenn = facetoface(f)
    ftop, _ = facetopoints(None, v, f, np.asarray([list(g) + [0.] for g in gamma]))
    # print("ftop:",ftop)
    # print("v2f:", v2f)
    labeled_verts = vert_knn(ftop, idxtofacenn, v2f, v, k=3, distance_threshold=.03)
    vert_w = []
    vert_debug =[]
    for i , _ in enumerate(v):
        vert_w.append(list(labeled_verts[i])[-1])
        vert_debug.append(list(labeled_verts[i]))
    # G = nx.Graph()
    # def add_tri(t):
    #     for x, y in [(0,1), (1,2), (2,0)]:
    #         # print(1-(vert_w[t[x]] + labeled_verts[t[y]])/2.)
    #         G.add_edge(t[x], t[y], weight= (vert_w[t[x]] + vert_w[t[y]])/2.)
    # for t in f:
    #     add_tri(t)
    #
    # paths = nx.single_source_dijkstra_path(G, 5)
    # print(paths[6])

    # export vert weights to file
    thresh = .03
    labeled_verts = list(enumerate(labeled_verts))
    filter(lambda x: x[1][-1] < thresh, enumerate(labeled_verts))

    a = sorted(labeled_verts, key = lambda x: x[1][0])[0][0]
    b = sorted(labeled_verts, key=lambda x: x[1][0])[-1][0]
    c = sorted(labeled_verts, key=lambda x: x[1][1])[0][0]
    d = sorted(labeled_verts, key=lambda x: x[1][1])[-1][0]
    e = sorted(labeled_verts, key=lambda x: x[1][2])[0][0]
    f = sorted(labeled_verts, key=lambda x: x[1][2])[-1][0]
    print(f"mesh {outfile} good seam startpts {a} {b} {c} {d} {e} {f}")
    with open(outfile, "w") as f:
        for i in vert_w:
            f.write(str(i) +"\n")
    np.savetxt(outfile+"debug.txt", vert_debug, delimiter=",")


    return None


def resample_directory(d, o, m, exclusion_list):
    precisions = []
    accs = []
    if not os.path.exists(o):
        os.mkdir(o)
    #inclusion_list = [136, 48, 138, 70, 80, 55, 75, 192]
    for i, f in enumerate(os.listdir(d)[:]):
        infile = os.path.join(d, f)
        num, _ = f.split(".")
        # print(num, d)
        numpad = num
        if int(num) in exclusion_list: # TODO change back
            continue
        if len(num) == 1:
            num = "00" + num
        elif len(num) == 2:
            num = "0" + num
        outfile = os.path.join(o,num+".csv")
        mfile = os.path.join(m, num + ".obj")
        # run_extract_seams(mfile, i)
        if try_catch:
            try:
                print("mfile:", mfile)
                mesh_knn(mfile, ct, infile, outfile, k)
            except:
                print(f"error for {mfile}")
        # extract_seams(None, mfile)
        # relabel_pts(new_pts, outfile, m, num, width= 0.03)

        #load_seams(int(numpad), m)

def load_seams(f,m):
    f = int(f)
    infile = f"/home/theresa/p/verts_seams/{f}.csv"
    gamma = genfromtxt(infile, dtype=np.int32, delimiter=",")
    num = str(f)
    if len(num) == 1:
        num = "00" + num
    elif len(num) == 2:
        num = "0" + num
    mfile = os.path.join(m, num + ".obj")
    v, _ = ig.read_triangle_mesh(mfile)
    verts = []
    for g in gamma:
        verts.append(v[g])
    outfile = f"/home/theresa/p/verts_points/{f}.csv"
    np.savetxt(outfile, np.asarray(verts), delimiter = ",")
    return verts


if __name__ == "__main__":
    # run_extract_seams("/home/theresa/p/data_v8_obj/310.obj", "/home/theresa/p/groundtruthseam/310.csv")
    meshes = "/home/theresa/Pointnet2_PyTorch/pointnet2/data/data_v1_obj/"
    #
    # for m in os.listdir(meshes):
    #     ind = int(m.split(".")[0])
    #     run_extract_seams(meshes + m, ind)

    v = str(sys.argv[1])

    train_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2" \
                           f"/model_output/{v}/points/preds"
    test_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2" \
                          f"/model_output" \
                          f"/{v}/points/eval_preds"
    train_output = f"/home/theresa/Pointnet2_PyTorch/pointnet2" \
                          f"/model_output/{v}/points/preds_resampled"
    test_output = f"/home/theresa/Pointnet2_PyTorch/pointnet2" \
                          f"/model_output/{v}/points/eval_preds_resampled"

    # train_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2" \
    #                        f"/data/{v}/"
    # train_output = f"/home/theresa/Pointnet2_PyTorch/pointnet2" \
    #                f"/data/{v}_resampled"

    if not os.path.isdir(train_output):
        os.makedirs(train_output)
    # if not os.path.isdir(test_output):
    #     os.makedirs(test_output)

    pool = mp.Pool(mp.cpu_count())
    print(f"processing {train_predicted_path} only")
    exclude = []
    results = [pool.apply_async(resample_directory, args=(path, train_output,
                                                    meshes, el)) for
               # comment back in!!
               path, el in [  (test_predicted_path, []),
                            (train_predicted_path, exclude +list(range(120,
                            500))) ,
                            (train_predicted_path, exclude + list(range(120)) +
                            list(range(240, 500))),
                            (train_predicted_path, exclude + list(range(240)) + list(
                                range(360,500))),
                            (train_predicted_path, exclude +list(range(360)))]]
    output = [p.get() for p in results]
    print(output)
    # Step 3: Don't forget to close
    pool.close()

    # print("TEST")
    # testp, testa = resample_directory(test_predicted_path,
    #                                   train_output,
    #                                   meshes,[48])# [68, 128, 125])
    # print("TRAIN")
    # trainp, traina = resample_directory(train_predicted_path, train_output,
    #               meshes, [])#[0, 5, 83, 112, 173, 191])