import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.cluster
import scipy.cluster.hierarchy

# import from this project
import data
import util

np.random.seed(data.RANDOM_SEED)
data_objs = [data.data_objs[i](n=data.N) for i in range(len(data.data_objs))]

# scipy single linkage
fig, axes = plt.subplots(1, len(data.data_objs), figsize=(10, 2))
for i, ax in enumerate(axes):
    data_obj = data_objs[i]
    X = data_obj.X
    y_hat = scipy.cluster.hierarchy.fclusterdata(
        X,
        t=data_obj.n_clusters,
        criterion='maxclust')
    util.scatter(X, ax, y_hat)
plt.tight_layout()
plt.savefig(
    os.path.join(
        '..', 'tex', 'figures',
        'agglomerative_clusters_{}_linkage.pdf'.format('minimum')))

# sklearn agglormerative
linkages = ['complete', 'average']
for linkage in linkages:
    fig, axes = plt.subplots(1, len(data.data_objs), figsize=(10, 2))
    for i, ax in enumerate(axes):
        data_obj = data_objs[i]
        X = data_obj.X
        cluster_obj = sklearn.cluster.AgglomerativeClustering(
            n_clusters=data_obj.n_clusters,
            linkage=linkage)
        y_hat = cluster_obj.fit_predict(X)
        util.scatter(X, ax, y_hat)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            '..', 'tex', 'figures',
            'agglomerative_clusters_{}_linkage.pdf'.format(linkage)))

# kmeans
fig, axes = plt.subplots(1, len(data.data_objs), figsize=(10, 2))
for i, ax in enumerate(axes):
    data_obj = data_objs[i]
    X = data_obj.X
    cluster_obj = sklearn.cluster.KMeans(
        n_clusters=data_obj.n_clusters)
    y_hat = cluster_obj.fit_predict(X)
    util.scatter(X, ax, y_hat)
plt.tight_layout()
plt.savefig(
    os.path.join(
        '..', 'tex', 'figures',
        'kmeans_clusters.pdf'))

# kmeans tesselation
fig, axes = plt.subplots(1, len(data.data_objs), figsize=(10, 2))
for i, ax in enumerate(axes):
    data_obj = data_objs[i]
    X = data_obj.X
    cluster_obj = sklearn.cluster.KMeans(
        n_clusters=data_obj.n_clusters)
    y_hat = cluster_obj.fit_predict(X)
    util.scatter(X, ax, y_hat)

    # do the background voronoi tesselation
    h = 0.02
    xx, yy = np.meshgrid(np.arange(-1.5, 1.5, h), np.arange(-1.5, 1.5, h))
    Z = cluster_obj.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.imshow(
           Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.tight_layout()
plt.savefig(
    os.path.join(
        '..', 'tex', 'figures',
        'kmeans_clusters_voronoi.pdf'))

# DBSCAN
params = [
    {'eps': 0.1, 'min_samples': 4},
    {'eps': 0.2, 'min_samples': 4},
    {'eps': 0.2, 'min_samples': 20}]
for params_dict in params:
    fig, axes = plt.subplots(1, len(data.data_objs), figsize=(10, 2))
    for i, ax in enumerate(axes):
        data_obj = data_objs[i]
        X = data_obj.X
        cluster_obj = sklearn.cluster.DBSCAN(
            eps=params_dict['eps'],
            min_samples=params_dict['min_samples'])
        y_hat = cluster_obj.fit_predict(X)
        util.scatter(X, ax, y_hat)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            '..', 'tex', 'figures',
            'DBSCAN_clusters_eps_{}_MinPts_{}.pdf'.format(
                params_dict['eps'], params_dict['min_samples'])))
