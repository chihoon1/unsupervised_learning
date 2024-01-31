'''
plotting functions that visualize data points for clusterings and etc.
'''
#import os, sys
#sys.path.insert(0, os.getcwd())
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_clustering(D, cluster_labels, **kwargs):
    '''
    Plot clustering of the dataset D. i-th index of cluster_labels is the clust label of i-th row(data) in D
    :param D: numpy 2d array dataset (expect 2 or 3-dimensional data)
    :param cluster_labels: numpy 1d array represent cluster label of data point in D
    :return: None
    '''
    dtype = kwargs.get('dtype', 'int32')  # data type of cluster labels. Default: integer
    fig = plt.figure()
    dim = 1 if len(D.shape) == 1 else D.shape[1]
    cluster_labels = cluster_labels.astype(dtype)
    if dim == 2:
        sns.scatterplot(x=D[:, 0], y=D[:, 1], hue=cluster_labels)
    elif dim == 3:
        ax = fig.add_subplot(projection='3d')
        num_clusters = len(np.unique(cluster_labels))
        for k in range(num_clusters):
            x = [D[i,0] for i in range(len(D)) if cluster_labels[i] == k]
            y = [D[i,1] for i in range(len(D)) if cluster_labels[i] == k]
            z = [D[i,2] for i in range(len(D)) if cluster_labels[i] == k]
            ax.scatter(x, y, z, label=k, depthshade=False)
        ax.legend(loc='upper right')
    else:
        raise Exception("Wrong Dimension Given! Dimension of data must be 2D or 3D.")
    plt.show()
    #fig.show()



