import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import sklearn.datasets

# import from this project
import util


class TwoGaussians:
    n_clusters = 2

    def __init__(self, n):
        mu = np.array([[-1, 0], [1, 0]])
        C = np.eye(2)
        C *= 0.04
        self.X = np.concatenate(
            [np.random.multivariate_normal(
                mu[i], cov=C, size=n) for i in range(mu.shape[0])], axis=0)
        self.y = np.concatenate([
            np.zeros(n),
            np.ones(n)])


class ThreeGaussians:
    n_clusters = 3

    def __init__(self, n):
        mu = np.array([[-1, -0.5], [1, -0.5], [0, 0.5]])
        C = np.eye(2)
        C *= 0.04
        self.X = np.concatenate(
            [np.random.multivariate_normal(
                mu[i], cov=C, size=n) for i in range(mu.shape[0])], axis=0)
        self.y = np.concatenate([
            np.zeros(n),
            np.ones(n),
            np.ones(n) * 2])


class ElongatedGaussians:
    n_clusters = 2

    def __init__(self, n):
        mu = np.array([[-0.5, 0], [0.5, 0]])
        C = np.eye(2)
        C[1, 1] = 40
        C *= 0.005
        self.X = np.concatenate(
            [np.random.multivariate_normal(
                mu[i], cov=C, size=n) for i in range(mu.shape[0])], axis=0)
        self.y = np.concatenate([
            np.zeros(n),
            np.ones(n)])


class TwoMoons:
    n_clusters = 2

    def __init__(self, n):
        self.X, self.y = sklearn.datasets.make_moons(n_samples=2 * n, noise=0.05)
        self.X /= 1.2
        self.X -= 0.4


class TwoOverlappingGaussians:
    n_clusters = 2

    def __init__(self, n):
        mu = np.array([[-0.7, 0], [0.7, 0]])
        C = np.eye(2)
        C *= 0.08
        self.X = np.concatenate(
            [np.random.multivariate_normal(
                mu[i], cov=C, size=n) for i in range(mu.shape[0])], axis=0)
        self.y = np.concatenate([
            np.zeros(n),
            np.ones(n)])


class DifferentDensities:
    n_clusters = 2

    def __init__(self, n):
        mu = np.array([[-0.8, 0], [0, 0]])
        C1 = np.eye(2)
        C1 *= 0.01
        C2 = np.eye(2)
        C2 *= 0.15
        self.X = np.concatenate([
            np.random.multivariate_normal(
                mu[0], cov=C1, size=n),
            np.random.multivariate_normal(
                mu[1], cov=C2, size=n)],
            axis=0)
        self.y = np.concatenate([
            np.zeros(n),
            np.ones(n)])


RANDOM_SEED = 1337
N = 80
data_objs = [TwoGaussians, ThreeGaussians, ElongatedGaussians,
             TwoMoons, TwoOverlappingGaussians, DifferentDensities]


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    data_objs = [data_objs[i](n=N) for i in range(len(data_objs))]

    fig, axes = plt.subplots(1, len(data_objs), figsize=(10, 2))
    for i, ax in enumerate(axes):
        data_obj = data_objs[i]
        X = data_obj.X
        util.scatter(X, ax)
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'tex', 'figures', 'sample_clusters.pdf'))

    fig, axes = plt.subplots(1, len(data_objs), figsize=(10, 2))
    for i, ax in enumerate(axes):
        data_obj = data_objs[i]
        X = data_obj.X
        util.scatter(X, ax, y=data_obj.y)
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'tex', 'figures', 'sample_clusters_true_assignment.pdf'))
