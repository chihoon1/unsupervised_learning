'''
Implementation of EM(Expectation-Maximization) clustering algorithm (Unsupervised Learning)
'''
import numpy as np
from scipy.stats import multivariate_normal
from plotting import plot_clustering

def EM_train(D, clusters_mean_list, clusters_covmat_list, prior_prob_list, **kwargs):
    '''
    Learn cluster's means and covariance matrices and prior probabilities from the dataset
    param: D=data(2d np array), cluster_mean_list=initial mu for each cluster(index=0:n-1 represent each cluster),
    clusters_covmat_list=initial covariance matrix for each cluster
    prior_prob_list=initial prior probabilities for each cluster
    return: tuple of clusters' means(list of np array), clusters' covariance matrices(list of np array), and
        clusters' prior probabilities(list of np array), and SSE(integer)
    '''
    epsilon = kwargs.get("epsilon", 0.001)  # float = preferred SSE for convergence criteria
    stop = kwargs.get("stop", 100)  # stop at this iteration if loop executes too long
    iteration = 0
    K = len(clusters_mean_list)
    curr_prev_SSE = float('inf')
    while curr_prev_SSE > epsilon and iteration < stop:
        iteration += 1
        clusters_weights = [[] for i in range(K)]  # 2d list(outer index: cluster id, inner elem: weight of xj)
        # expectation step
        for j in range(D.shape[0]):  # going through all data points
            xj = D[j]
            denom = 0
            for i in range(2*K):  # going through all clusters
                if i < K:
                    # totaling up the denominator of posterior probability
                    # summing up all f(xj | mu_i, sigma_i)*P(Ci) for all clusters i=1:k
                    observed_prob = multivariate_normal.pdf(xj, mean=clusters_mean_list[i],
                                                        cov=clusters_covmat_list[i], allow_singular=True)
                    denom += observed_prob*prior_prob_list[i]
                else:
                    # computing posterior probability
                    ind = i % K
                    # invoking f(x | mu, sigma) function
                    numerator1 = multivariate_normal.pdf(xj, mean=clusters_mean_list[ind],
                                                        cov=clusters_covmat_list[ind], allow_singular=True)
                    clusters_weights[ind].append(numerator1 * prior_prob_list[ind] / denom)
                #print(f"{i}'s denom: {denom}'")  # debugging purpose
                #print(f"{i}'s weight:\n{clusters_weights}")  # debugging purpose
        # print(f"weights:\n{clusters_weights}")  # debugging purpose
        # maximization step
        temp_cluster_mean = []
        for i in range(K):  # iterate through each cluster
            # update cluster mean for each cluster
            wi_vector = np.array(clusters_weights[i])
            denom = np.inner(wi_vector, np.array([1 for x in range(D.shape[0])]))
            updated_mu_i = D.T@wi_vector/denom
            temp_cluster_mean.append(updated_mu_i)
            # update cluster covariance matrix for each cluster
            numerator = np.zeros((D.shape[1], D.shape[1]))
            for j in range(D.shape[0]):
                xj_centered_mu_i = D[j] - temp_cluster_mean[i]  # use the updated mean to compute centered xj
                xj_centered_mu_i.shape = (D.shape[1], 1)
                numerator += wi_vector[j]*(xj_centered_mu_i@xj_centered_mu_i.T)
            updated_sigma_i = numerator/denom
            clusters_covmat_list[i] = updated_sigma_i  # inplace update by replacing previous iteration parameter
            # update prior prob for each cluster
            updated_prior_prob = denom / D.shape[0]  # for updating prior probability, we use <wi,1> in the numerator
            prior_prob_list[i] = updated_prior_prob  # inplace update by replacing previous iteration parameter
        print(f"iteration {iteration}'s temp_cluster_mean:\n{temp_cluster_mean}")  # debugging
        print(f"iteration {iteration}'s clusters_covmat_list:\n{clusters_covmat_list}")  # debugging
        print(f"iteration {iteration}'s prior_prob_list:\n{prior_prob_list}")  # debugging
        # print(f"iteration {iteration}'s Sum of prior prob: {sum(prior_prob_list)}")  # debugging purpose
        # I'm using the Frobenius norm because the Frobenius norm is the square root of sum of abs(entry)^2
        # but entry^2 = (-entry)^2 or (entry)^2. Thus, this would give the same result as SSE induced by l-2 norm
        curr_prev_SSE = np.linalg.norm(np.array(temp_cluster_mean) - np.array(clusters_mean_list), ord='fro')**2
        clusters_mean_list = temp_cluster_mean
        print(f"iteration {iteration}'s SSE between current and previous iteration: {curr_prev_SSE}")  # debugging
    return clusters_mean_list, clusters_covmat_list, prior_prob_list, curr_prev_SSE


def EM_clustering(D, clusters_mean_list, clusters_covmat_list, prior_prob_list, **kwargs):
    # Cluster data based on the given clusters means, clusters covariance matrices, and cluster prior probabilities
    '''
    :param D: data(2d np array)
    :param clusters_mean_list: list of mean of each cluster(index=0:n-1 represent each cluster),
    :param clusters_covmat_list: list of covariance matrix for each cluster
    :param prior_prob_list: list of prior probabilities for each cluster
    :return: array containing assigned clusters for each data point in the dataset (same order appears in the dataset)
    '''
    cluster_labels = np.zeros(len(D))  # array containing assigned clusters for each data point
    for i, data_point in enumerate(D):
        max_prob, assigned_cluster = 0, -1
        # data point is assigned to a clust by argmax f(xj | mu_i, sigma_i)*P(Ci) for all clusters i=1:k
        for k in range(len(clusters_mean_list)):
            # compute the probability of observing data point with given current cluster's mean and covariance
            prob = multivariate_normal.pdf(data_point, mean=clusters_mean_list[k],
                                                        cov=clusters_covmat_list[k], allow_singular=True)
            prob = prob * prior_prob_list[k]
            if prob > max_prob:  # argmax step
                max_prob = prob
                assigned_cluster = k
        cluster_labels[i] = assigned_cluster
    return cluster_labels


if __name__ == '__main__':
    # Test case
    X1 = [1.5, 2.2, 3.9, 2.1, 0.5, 0.9, 2.7, 2.5, 2.8, 0.1]
    X2 = [4.5, 1.5, 3.4, 2.9, 3.2, 4.3, 2.1, 3.5, 4.9, 3.7]
    D = np.column_stack((X1, X2))
    # Parameters initialization
    k = 3
    mu_list = [np.array([0.5, 4.2]), np.array([2.4, 1.6]), np.array([3, 3.2])]
    covmat_list = []
    for i in range(3):
        covmat_list.append(np.identity(D.shape[1]))
    prior_probabilities = [1 / 3, 1 / 3, 1 / 3]
    eps = 0.001
    # EM steps
    clusters_mu, clusters_covm, clusters_prior_pr, SSE = EM_train(D, mu_list, covmat_list, prior_probabilities, epsilon=eps)
    # clustering
    cluster_labels = EM_clustering(D, clusters_mu, clusters_covm, clusters_prior_pr)
    plot_clustering(D, cluster_labels)

    # 3D data
    D = np.random.normal(size=(9,3)) * 4
    # Parameters initialization
    k = 2
    mu_list = [np.random.normal(size=3) * 4, np.random.normal(size=3) * 4]
    covmat_list = []
    for i in range(2):
        covmat_list.append(np.identity(D.shape[1]))
    prior_probabilities = [1 / 3, 1 / 3]
    eps = 0.001
    # EM steps
    clusters_mu, clusters_covm, clusters_prior_pr, SSE = EM_train(D, mu_list, covmat_list, prior_probabilities,
                                                                  epsilon=eps)
    # clustering
    cluster_labels = EM_clustering(D, clusters_mu, clusters_covm, clusters_prior_pr)
    plot_clustering(D, cluster_labels)
