'''
Implementation of K-means clustering algorithm (Unsupervised Learning)
'''

import numpy as np
from plotting import plot_clustering


def k_means_clustering(D, centroid_list, **kwargs):
    '''
    param: D = data set
    param: centroid_list = initial list of centroid(list of np.arrys)
    param: epsilon = preferred sum of squared difference between
            i-th and i-1-th iteration centroids for convergence
    return: tuple containing cluster assignments, centroids,
            SSE of clusters, and iteration at convergence
    '''
    # preferred sum of squared difference between i-th and i-1-th iteration centroids for convergence
    epsilon = kwargs.get('epsilon', 0)

    iteration = 0
    sum_of_squared = float('inf')  # arbitrary number greater than epsilon for first iteration
    # iteration stopping criteria when no change in clusters between iterations
    while (sum_of_squared > epsilon):
        iteration += 1
        # Cluster Assignment for each xj in D
        temp_clusters_list = []
        for i in range(len(centroid_list)):
            temp_clusters_list.append(set())  # can't do []*len() because it copies the same set object
        temp_centroid_list = [0]*len(centroid_list)
        for j in range(D.shape[0]):
            xj = D[j,:]  # xj is a sample data point(row of D)
            min_distance = float('inf')
            for i in range(len(centroid_list)):
                mu_i = centroid_list[i]# mu_i is the centroid of cluster_i
                # the closest cluster is
                # min(squared of l-2 norm of (xj minus each centroid))
                dist_xj_mu_i = np.linalg.norm(xj - mu_i)**2
                if dist_xj_mu_i < min_distance:
                    cluster_id = i
                    min_distance = dist_xj_mu_i
            #print("{cluster_id} and dist: {min_distance}".format(cluster_id=cluster_id, min_distance=min_distance))  # debugging
            temp_clusters_list[cluster_id].add(j)
        # update centroid
        for i in range(len(temp_clusters_list)):
            cluster = temp_clusters_list[i]
            total = np.array([0.0]*D.shape[1])
            for j in cluster:
                total += D[j,:]
            temp_centroid_list[i] = total / len(cluster)
            #print(f"updated cluster: {cluster}")  # debugging
        #print(f"updated centroid: {temp_centroid_list}")  # debugging
        # Measure the sum of squared difference in centroids between the current iteration and the previous iteration
        # if the sum of squared difference in centroies = 0, it implies no change in centroids and in clusters
        sum_of_squared = 0
        for i in range(len(centroid_list)):
            se = np.linalg.norm(temp_centroid_list[i] - centroid_list[i])**2
            sum_of_squared += se
        centroid_list = temp_centroid_list
        clusters_list = temp_clusters_list
    # computing the SSE of clusters
    SSE = 0
    for i in range(len(centroid_list)):
        for j in clusters_list[i]:
            SSE += np.linalg.norm(D[j,:] - centroid_list[i])**2
    return clusters_list, centroid_list, SSE, iteration



if __name__ == '__main__':
    # test the k-means function
    X1 = np.array([0, 0, 2, 5, 5])
    X2 = np.array([2, 0, 0, 0, 2])
    D = np.column_stack((X1, X2))
    centroid_list = [D[0,:], D[1, :]]
    clusters_list, centroid_list, SSE, iteration = k_means_clustering(D, centroid_list)
    for i in range(len(clusters_list)):
        print(f"Cluster {i + 1}")
        for j in clusters_list[i]:
            print(f"Data point in Cluster {i + 1}:\nx{j + 1} = {D[j, :]}")
    for i in range(len(centroid_list)):
        print(f"Cluster {i + 1}'s centroid: {centroid_list[i]}")
    print(f"SSE of clusters: {SSE}")
    print(f"Converged in {iteration}-th iteration")

    # test with different centroids
    centroid1 = (D[0, :] + D[1, :] + D[2, :] + np.array([2, 2])) / 4  # imaginary vertex (2,2)
    centroid2 = (D[3, :] + D[4, :]) / 2
    new_centroid_list = [centroid1, centroid2]
    clusters_list, centroid_list, SSE, iteration = k_means_clustering(D, new_centroid_list)
    for i in range(len(clusters_list)):
        print(f"Cluster {i + 1}")
        for j in clusters_list[i]:
            print(f"Data point in Cluster {i + 1}:\nx{j + 1} = {D[j, :]}")
    for i in range(len(centroid_list)):
        print(f"Cluster {i + 1}'s centroid: {centroid_list[i]}")
    print(f"SSE of clusters: {SSE}")
    print(f"Converged in {iteration}-th iteration")
    print(clusters_list)

    # make cluster labels for data points
    cluster_labels = np.zeros(len(D))
    for i in range(len(clusters_list)):
        for data_point in clusters_list[i]:
            cluster_labels[data_point] = i
    plot_clustering(D, cluster_labels)
