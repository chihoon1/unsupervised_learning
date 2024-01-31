'''
Implementation of singlely-linked hierarchy clustering
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def single_link_hierarchy_clustering(adj_mat: list, k=1, **kwargs):
    # perform single linked hierarchical clustering
    # points: space of points seen in the data set. adj_mat: adjacency matrix of points
    # k is desired # of clusters. By default, it's 1
    # return cluster sets for each depth(list of sets) and shortest distance log(list) as a tuple
    temp_mat = []
    for j in range(len(adj_mat)):  # forming full symmetric matrix for adjacency matrix
        row = []
        for i in range(len(adj_mat)):
            if i == j: row.append(0)
            elif j < i: row.append(adj_mat[j][i])  # lower triangular part
            else: row.append(adj_mat[i][j])
        temp_mat.append(row)
    adj_mat = np.array(temp_mat)
    # cluster_levels contain sets of clusters for all levels in hierarchy. each level is represented by set
    #cluster_levels = [[frozenset(x) for x in points]]
    cluster_levels = [[frozenset([x]) for x in range(len(adj_mat))]]
    i = 0
    shortest_dist_log = []  # contain the shortest distance connected two clusters at each level
    while len(cluster_levels[i]) > k:  # merge the closest pair of clusters until hits k # of clusters
        shortest_dist = float('inf')
        for l in range(len(cluster_levels[i])):
            for m in range(len(cluster_levels[i])):
                if (l != m) and (adj_mat[l,m] < shortest_dist):
                    close_clusters = [(cluster_levels[i][l],l),(cluster_levels[i][m],m)]  # list of the closest clusters
                    shortest_dist = adj_mat[l,m]
        shortest_dist_log.append(shortest_dist)
        print(f"Iter {i} shortest distance: {shortest_dist}")
        # lance-williams formula of single link to update the merged cluster's distance to other clusters
        row_l = adj_mat[close_clusters[0][1], :]
        row_m = adj_mat[close_clusters[1][1], :]
        updated_dist = (1/2*row_l + 1/2*row_m - 1/2*np.absolute(row_l-row_m))
        # print(f"new distance: {updated_dist}")  # debugging purpose
        adj_mat[close_clusters[0][1], :] = updated_dist  # updated distance will be stored in l-th row(cluster l)
        adj_mat = np.delete(adj_mat, close_clusters[1][1], axis=0)  # drop row m (cluster m)
        adj_mat = np.delete(adj_mat,close_clusters[1][1], axis=1)  # drop column m (cluster m)
        adj_mat[:, close_clusters[0][1]] = adj_mat[close_clusters[0][1], :]  # maintain symmetric for col l and row l
        # join the cluster i and cluster j with the shortest distance and exclude each one of them from the clusters
        new_cluster = close_clusters[0][0].union(close_clusters[1][0])
        level_i_clusters = cluster_levels[i][:]
        level_i_clusters[close_clusters[0][1]] = new_cluster
        level_i_clusters.pop(close_clusters[1][1])
        cluster_levels.append(level_i_clusters)
        print(f"{i} iter:\n{adj_mat}")  # debugging purpose
        i += 1
    return cluster_levels, shortest_dist_log  # clusters at each level of hierarchy


def convert_to_linkage_matrix(cluster_hierarchy, shortest_dist_log):
    # convert sets of clusters at each level in hierarchy to scipy hierarchy linkage matrix
    # param cluster_hierarchy: 2d container
    # param adj_mat: numpy 2d array
    # return a scipy linkage matrix. shape: (l, 4) where l == depth in the hierarchy
    linkage_mat = np.zeros((len(cluster_hierarchy)-1, 4))
    cluster_names_dict = {}
    for i, cluster in enumerate(cluster_hierarchy[0]):
        cluster_names_dict[cluster] = i
    for i in range(1, len(cluster_hierarchy)):
        # find the newly formed cluster at level i
        curr_clusters = set(cluster_hierarchy[i])  # cluster set at current depth level
        prev_clusters = set(cluster_hierarchy[i - 1])  # cluster set at previous(lower) depth level
        newly_formed_c = tuple(curr_clusters.difference(prev_clusters))[0]  # newly formed cluster added at current level
        # two subclusters formed the newly formed cluster
        subcluster1, subcluster2 = tuple(prev_clusters.difference(curr_clusters))
        cluster_names_dict[newly_formed_c] = len(cluster_hierarchy[0]) + i - 1
        # col0 and col1: two subclusters' names
        linkage_mat[i-1, 0] = cluster_names_dict[subcluster1]
        linkage_mat[i-1, 1] = cluster_names_dict[subcluster2]
        linkage_mat[i-1, 2] = shortest_dist_log[i-1]  # distance between subcluster1 and 2
        linkage_mat[i-1, 3] = cluster_names_dict[newly_formed_c]  # newly formed cluster name
    return linkage_mat

if __name__ == '__main__':
    points = ['A', 'B', 'C', 'D', 'E', 'F']
    # 5*5 matrix(list of lists) of single link distance. row omitting F(last row) and column omitting A(first column)
    adjacency_mat = [
        [0, 7, 57, 36, 42, 32],
        [0, 0, 49, 29, 35, 23],
        [0, 0,  0, 22, 14, 25],
        [0, 0,  0,  0,  5, 10],
        [0, 0,  0,  0,  0, 11],
        [0, 0,  0,  0,  0,  0],
    ]
    cluster_hierarchy, shortest_dist_log = single_link_hierarchy_clustering(adjacency_mat)
    for level, cluster in enumerate(cluster_hierarchy):
        print(f"depth {len(cluster_hierarchy) - level}: {cluster}")

    linkage_mat = convert_to_linkage_matrix(cluster_hierarchy, shortest_dist_log)
    dendrogram(linkage_mat, labels=points)
    plt.title("Singly Linked Hierarchical Clustering Plot")
    plt.show()

