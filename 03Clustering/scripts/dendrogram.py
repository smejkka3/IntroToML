import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import os


np.random.seed(1337)
mu = np.array([[-1, 0], [1, 0], [0, 0]])
C = np.eye(2)
C[1, 1] = 15
C *= 0.005
X = np.concatenate([np.random.multivariate_normal(mu[i], cov=C, size=9) for i in [0, 1, 2]], axis=0)

for distance in ['euclidean', 'squared_euclidean']:
    if distance == 'euclidean':
        scipy_distance = 'euclidean'
    elif distance == 'squared_euclidean':
        scipy_distance = 'sqeuclidean'
    else:
        raise Exception('Unknown distance')
    linkage = hierarchy.linkage(X, metric=scipy_distance)

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(X[:, 0], X[:, 1])
    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([-1.5, 1.5])
    axes[0].set_xlabel(r'$x_1$')
    axes[0].set_ylabel(r'$x_2$')
    hierarchy.dendrogram(linkage, ax=axes[1])
    axes[1].set_xlabel('sample index')
    axes[1].set_ylabel('distance')
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'tex', 'figures', 'dendrogram_{}.pdf'.format(distance)))
