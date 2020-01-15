import sys
sys.path.append("/home/theresa/libigl/python")
import pyigl as old_igl
import igl as ig
from iglhelpers import *
import numpy as np
from numpy import genfromtxt
import os

test_run = False

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
    print([i.face for i in idxtofacenn[3].neighbors])
    return idxtofacenn

def facetopoints(points, v, f, sampled_points=None):
    if points:
        gamma = genfromtxt(points, dtype=np.float64, delimiter=",")
        if test_run:
            gamma = np.asarray([[0.,0.1,0.,0], [0.,0.5,0.,1],  [0., 0.6, 0., 1]], dtype=np.float64)
        pointsonly = np.array([list(g[:-1]) for g in gamma])

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
        if fi in ftop:
            ftop[fi].append(z)
        else:
            ftop[fi] = [z]
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
            point_faces.append(face)
            verts = [v[i] for i in f[face]]
            pt = sum([float(b[i]) * verts[i] for i in range(len(verts))])
            ws_points.append(pt)

        opposite_points = np.array([[-1 * x, y, z] for x, y, z in ws_points])
        _, opp_prims, _ = ig.point_mesh_squared_distance(opposite_points, v, f)
        opposite_point_faces = opp_prims.astype(np.int64)
        for idx, _ in enumerate(ws_points):
            opposite_point_face = opposite_point_faces[idx]
            face = point_faces[idx]
            pointstoface[idx] = [face, opposite_point_face]
        if test_run:
            ws_points = np.asarray([[0., 0.1, 0.], [0., 1.5, 0.]], dtype=np.float64)
        labeled_verts = vert_knn(ftop, idxtofacenn, pointstoface, ws_points, k=k)
        # _, pointstoface = facetopoints(None, v, f, sampled_points)
        # gamma = np.asarray([[0., 0.1, 0.], [0., 0.5, 0.]], dtype=np.float64)
        for vert in labeled_verts:
            if vert[3] < np.random.uniform(0,1):
                pts.append(vert)
            # pts.append(vert[:])
        print("points sampled:", len(pts))
    pts = pts[:n_pts]
    np.savetxt(output, np.asarray(pts), delimiter=",")


def vert_knn(ftop, idxtofacenn, v2f, sampled_points, k=5,seam_threshold=0, distance_threshold=.01):

    labeled_verts = []
    for i, verts in enumerate(sampled_points):
        # Check my prim first
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

            if len(points) < k-len(close_points):
                close_points += points
            else:
                # add all the points on the fringe to the points in sorted order
                points = sorted(points, key=lambda x: np.linalg.norm(np.asarray(x[:-1])- np.asarray(verts)))
                close_points += points[:k-len(close_points)]
            new_fringe = []

            for p in fringe:
                for n in idxtofacenn[p].neighbors:
                    if n.face not in visited_prims:
                        new_fringe.append(n.face)
                        visited_prims.append(n.face)
            fringe = new_fringe

        tot = sum([np.linalg.norm(np.asarray(x[:-1]) - np.asarray(verts)) for x in close_points])
        weights = []

        # print("for", verts, tot)
        if len(close_points)== 0:
            prob = 1
        elif len(close_points) == 1:
            prob = close_points[0][3]
        else:
            distances = []
            for pt in close_points: # TODO weight special for sparse pts or max radius

                # print("for", pt, )
                d = np.linalg.norm(np.asarray(pt[:-1])- np.asarray(verts))
                # print("distance", d, "d/t", (d / float(tot)), "weight=", 1-(d/float(tot)))
                distances.append(d)
                weights.append(1. - (d / float(tot)))
            if min(distances) > distance_threshold:
                prob = 1.
            else:

                # print("total weight", sum(weights))
                # print("-----")
                weights /= sum(weights)
                prob = 0.
                for w, pt in zip(weights, close_points): # TODO weight special for sparse pts or max radius
                    prob += w * pt[3]
                # seam threshold: scores lower mapped to zero
                # if weight < seam_threshold:
                #     weight = 0.
        verts = verts.tolist()
        verts.append(prob)
        labeled_verts.append(np.asarray(verts))
    if test_run:
        print("labeled_verts", labeled_verts)
    return labeled_verts

def relabel_pts(random_points, ptfile, m, num, width):
    # load points
    # unlabeled_points = genfromtxt(ptfile, dtype=np.float64, delimiter=",")

    unlabeled_points = []
    for r in random_points:
        r = list(r)
        unlabeled_points.append(r[:3])
    unlabeled_points =np.array(unlabeled_points)
    if test_run:
        unlabeled_points = np.array([[0, 0.1, 0], [0, 0.2, 0]])
    # attribute points to faces
    num = str(num)
    if len(num) == 1:
        num = "00" + num
    elif len(num) == 2:
        num = "0" + num
    mfile = os.path.join(m, num + ".obj")
    v, f = ig.read_triangle_mesh(mfile)
    f = f.astype(np.int32)
    if test_run:
        v = np.asarray([[0, 2, 0], [1, 1, 0], [-1, 1, 0], [1, 2, 0], [-1, 3, 0], [0, 0, 0]], dtype=np.float64)
        f = np.asarray([[3, 2, 6], [3, 1, 2], [1, 4, 2], [1, 5, 4]], dtype=np.int32)
        f = np.asarray([[i - 1 for i in p] for p in f], dtype=np.int32)
    _, prims, point_verts = ig.point_mesh_squared_distance(unlabeled_points, v, f)

    # find where pts are
    bary = []
    for pt,vert in zip(unlabeled_points,prims):
        v1, v2, v3 = [v[i] for i in f[vert]]


        def toArray(p, d= np.float64):
            p = list(p)
            p = np.array([p], dtype=d)
            return p

        # print("b")
        b = ig.barycentric_coordinates_tri(toArray(pt, np.float64), toArray(v1), toArray(v2), toArray(v3))
        bary.append(b)

    seam_verts = load_seams(num, m)
    if test_run:
        seam_verts = [4]
    # label face verts with geodesic distances
    point_verts_dict = {}
    for p in prims:
        for i in f[p]:
            if i not in point_verts_dict:
                point_verts_dict[i] = 0
    pt_list = list(point_verts_dict.keys())
    print("num_verts, keys:", len(v), len(pt_list))
    # d = ig.exact_geodesic(v, f,   toArray(seam_verts, np.int32), toArray(pt_list, np.int32),None, None)
    # print("c")
    pt_list1 = pt_list[:int(len(pt_list)/2)]
    pt_list2 = pt_list[int(len(pt_list)/2):]

    d1 = ig.exact_geodesic(v, f, np.array(seam_verts, dtype=np.int32), np.array(pt_list1, dtype=np.int32), None, None)
    d2 = ig.exact_geodesic(v, f, np.array(seam_verts, dtype=np.int32), np.array(pt_list2, dtype=np.int32), None, None)
    print("distances:", len(d1)+len(d2))
    for di, pi in zip(d1, pt_list1):
        point_verts_dict[pi] = di
    for di, pi in zip(d2, pt_list2):
        point_verts_dict[pi] = di
    # label points with interpolated distances
    labeled_points = []
    for pt, prim, b   in zip(unlabeled_points, prims, bary):
        l = sum([float(b[i]) * point_verts_dict[vidx] for i, vidx in enumerate(f[prim])])
        pt = list(pt)
        pt.append(l)
        labeled_points.append(pt)
    if test_run:
        print("source:", toArray(pt_list, np.int32), "dest", toArray(seam_verts, np.int32))
        print("dist:", d, "pts", pt_list)
        print(labeled_points)

    np.savetxt(ptfile, np.array(labeled_points), delimiter=", ")


def resample_directory(d, o, m, exclusion_list):
    precisions = []
    accs = []
    if not os.path.exists(o):
        os.mkdir(o)
    #inclusion_list = [136, 48, 138, 70, 80, 55, 75, 192]
    for i,f in enumerate(os.listdir(d)[:]):
        infile = os.path.join(d, f)
        num, _ = f.split(".")
        print(num)
        numpad = num
        if int(num) in exclusion_list: # TODO change back
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
        mesh_knn(mfile, 10000, infile, outfile, k=7)
        # relabel_pts(new_pts, outfile, m, num, width= 0.03)

        #load_seams(int(numpad), m)
    # print("precision: ",p)
        # print("accuracy: ", a)
        # except:
        #     pass
    return precisions, accs

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
    np.savetxt(outfile,np.asarray(verts), delimiter = ",")
    return verts


if __name__ == "__main__":
    meshes = "/home/theresa/p/datav5_obj"

    v = str(sys.argv[1])
    c = v.split("/")
    c = c[-2] + c[-1]
    v = c
    train_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/{v}/train_guesses"
    test_predicted_path = f"/home/theresa/Pointnet2_PyTorch/pointnet2/generated_shapes_debug/{v}/test_guesses"
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
