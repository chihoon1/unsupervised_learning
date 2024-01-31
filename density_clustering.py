'''
Implementation of Density Clustering algorithm (unsupervised learning)
'''
import numpy as np
from plotting import plot_clustering


def neighbours_of_x(D, x_idx: int, radius: int, **kwargs):
    # param D: dataset in np ndarray
    # param x_idx: a row index of a data point x
    # param radius: radius of a neighborhood ball
    neighbours = []
    for j in range(D.shape[0]):
        if j == x_idx: continue  # neighbors exclude point x itself
        dist = np.linalg.norm(D[x_idx] - D[j])  # distance between x and xj (l2 norm=Euclidean dist)
        if dist <= radius:  # add to neighbors list if distance smaller than radius
            neighbours.append(j)
    return neighbours


def density_connected(D, x_i: int, radius: int, pts_cluster_id: dict, k: int, border_pts: set, **kwargs):
    # Goal: assign cluster id k to any points reachable by xi and find paths to those reachable points
    # param D: dataset in np ndarray
    # param x_i: row index of a core point
    # param radius: radius of a neighborhood ball
    # param pts_cluster_id: assigned cluster of points
    # param k: current cluster id
    # param border_pts: a set of border points
    # return: border_pts(set) and path from directly reachable neighbor to core point
    #   not returning pts_cluster_id because dictionary is a mutable object
    incl = kwargs.get("inclusive", 0)  # indicate whether neighborhood ball includes xi itself. includes=1, excludes=0
    minpts = kwargs.get("minpts", 3)  # threshold to be considered as a core point
    if pts_cluster_id[x_i] == None: pts_cluster_id[x_i] = k  # assign cluster id to a point xi
    xi_neighbours = neighbours_of_x(D=D, x_idx=x_i, radius=radius)  # get the neighbors of a point xi
    paths = []
    if len(xi_neighbours) + incl >= minpts:  # xi is a core point, so jump to neighboring points allowed
        for point_j in xi_neighbours:
            if pts_cluster_id[point_j] != k:  # to prevent infinite loop by avoiding revisiting a point
                # recursive calls to find neighbors of xi's neighbor
                package = density_connected(D, point_j, radius, pts_cluster_id, k, border_pts, minpts=minpts, inclusive=incl)
                border_pts, path_from_recursion = package
                for i in range(len(path_from_recursion)):  # collect all paths from a neighbor to its neighbors
                    paths.append(path_from_recursion[i])
    else:  # xi is a border point, so can't move to neighboring points
        border_pts.add(x_i)
        paths.append([])
    for i in range(len(paths)):  # add xi(starting point) itself into a path
        paths[i].append(x_i)
    return border_pts, paths


def density_clustering(D, radius: int, **kwargs):
    # param D: data set in np ndarray
    # param radius: radius of neighborhood ball
    # param: include_itself_in_Ne(include itself for counting points within its radius)
    # return: points_cluster_id(dict), core_points(list), border_pts(set), noise_points(set), k(# of clusters)
    incl = kwargs.get("inclusive", 0)  # indicate whether neighborhood ball includes xi itself. includes=1, excludes=0
    minpts = kwargs.get("minpts", 3)  # threshold to be considered as a core point
    core_points = []  # list of all core points in the data set
    noise_points = set()  # set of all noise points
    points_cluster_id = {}  # dictionary where key=point index and value=its assigned cluster id
    for i in range(D.shape[0]):
        xi_neighbours = neighbours_of_x(D=D, x_idx=i, radius=radius)  # get neighbours of xi
        points_cluster_id[i] = None  # initialization. cluster id will be updated in next loop
        if len(xi_neighbours) + incl >= minpts: core_points.append(i)  # xi is a core point
        # initially, non-core points are assumed to be noise, but will be reclassified later
        else: noise_points.add(i)
    k = 0  # cluster id starts from 0
    border_pts = set()
    for i in range(len(core_points)):
        if points_cluster_id[i] != None: continue  # already have cluster assigned
        # assign the same cluster id for any points reachable to a core point and find border points
        density_connected(D, core_points[i], radius, points_cluster_id, k, border_pts, minpts=minpts, inclusive=1)
        k += 1  # cluster assignment for next cluster
    noise_points = noise_points.difference(border_pts)  # find the noise points
    return points_cluster_id, core_points, border_pts, noise_points, k-1


if __name__ == '__main__':
    points_notation = [char for char in 'abcdefghijklmnopqrstuvwxyz']
    x = [4, 5, 10, 3, 6, 9, 8, 13, 1, 2, 6, 8, 12, 13, 8, 14, 6, 7, 10, 4, 3, 9, 11, 15, 13, 13]
    y = [10, 11, 8, 8, 8, 5, 3, 8, 7, 7, 7, 7, 8, 7, 6, 6, 3, 5, 4, 3, 6, 2, 3, 4, 3, 2]
    D = np.array([x, y]).T
    radius = 2
    minpts = 3
    package = density_clustering(D, radius, minpts=minpts, inclusive=1)
    points_cluster_id, core_points, border_pts, noise_points, k = package
    # prettify output
    clusters = [[] for i in range(k + 1)]  # list of clusters(set containing its elements/points)
    cluster_labels = ['' for i in range(len(points_notation))]
    for key, value in points_cluster_id.items():
        if value is not None: clusters[value].append(points_notation[key])
        cluster_labels[key] = str(value) if value is not None else 'noise'
    for i in range(len(clusters)):
        print(f"Cluster {i}: {clusters[i]}")

    # plot clusters
    plot_clustering(D, cluster_labels=np.array(cluster_labels), dtype='str')

    # find paths if a path from r to z exists
    placeholder = {}  # placeholder for pts_cluster_id argument in density_connected function
    for i in range(D.shape[0]):
        placeholder[i] = None
    _, path = density_connected(D, points_notation.index('r'), radius, placeholder,
                                    k=0, border_pts=set(), inclusive=1, minpts=minpts)
    path_rz = []
    for i in range(len(path)):
        in_char = []
        for j in range(len(path[i])-1, -1, -1):
            in_char.append(points_notation[path[i][j]])
            if points_notation[path[i][j]] == 'z':
                path_rz.append(in_char)
                break
    print(f"The path from r to z is {path_rz}")

