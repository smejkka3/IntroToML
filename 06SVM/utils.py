import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch as tr
from imageio import imread
from scipy.spatial.distance import cdist, pdist
import glob
from sklearn.decomposition import PCA as sklPCA
from sklearn.datasets import make_moons

datadir = '/home/space/datasets/'
# datadir = '/Users/jack/data/'


def linear_kernel(X, Y, device='cpu'):
    return tr.tensor(X,device=device).mm(tr.tensor(Y.T,device=device)).numpy()


def get_parity_data(n_dims=15):
    A = [[0, 1]]*n_dims
    X = np.array(list(itertools.product(*A))).astype(np.float)
    Y = parity(X)
    return X, Y


def get_image_data(n_train=800, n_test=800, classes=['chaparral', 'christmas_tree_farm']):
    n_samples = n_train + n_test
    if n_samples > 1600:
        print("Only 1600 data points available. Data sets are split with users train/test ratio")
        n_train = int(n_train * (1600/n_samples))
        n_test  = int(n_test  * (1600/n_samples))
        print("n_train = %d, n_test = %d"%(n_train, n_test))
        n_samples = n_train + n_test
    if len(classes) > 2:
        print("For this exercise, only two classes are admissible. Returning first two classes")
    files = []
    Y = []
    for c, label in zip(classes, [-1, 1]):
        files += glob.glob(datadir + 'PatternNet/' + c + '/*.jpg')
        Y += [label]*800

    I = np.random.choice(1600, size=n_samples, replace=False)
    X = np.array([imread(f).flatten() for f in np.array(files)[I]]) / 255
    Y = np.array(Y)[I]

    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def get_2d_data(n_train=100, n_test=100):
    X, y = make_moons(n_samples=n_train+n_test, shuffle=True, noise=.3)
    y[y == 0] -= 1
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def get_2d_structured(n_train=100, n_test=100):
    n_grid = np.ceil((n_train+n_test)**.5).astype(np.int) + 2
    n_samples = n_grid**2
    xx, yy = np.meshgrid(np.linspace(0, n_grid-1, n_grid), np.linspace(0, n_grid-1, n_grid))
    X = np.c_[xx.ravel(), yy.ravel()]
    y = ((X.sum(1).astype(np.int) % 2) == 0).astype(np.int)
    y[y == 0] = -1

    X = X + .9*(np.random.rand(*X.shape) - .5)
    C = np.linalg.inv([[.9, .4], [-.2, 1.5]])
    X = X.dot(C)
    I = np.random.permutation(n_samples)
    X, y = X[I], y[I]
    return X[:n_train], y[:n_train], X[n_train:n_train+n_test], y[n_train:n_train+n_test]


def plot_2d(X, y, svm=None):
    xmin, xmax = X.min(), X.max()
    rng = xmax-xmin
    xlim = np.array([-rng, rng])*.6 + X.mean(0)[0]
    ylim = np.array([-rng, rng])*.6 + X.mean(0)[1]

    if svm is not None:
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300), np.linspace(ylim[0], ylim[1], 300))
        XX = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(XX).reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=20, zorder=1, cmap='coolwarm', vmin=-abs(Z).max(), vmax=abs(Z).max())
        plt.contour(xx, yy, Z, levels=[0], zorder=2, colors='k', alpha=.5, linewidths=1)

        plt.scatter(svm.X[:,0], svm.X[:,1], c='limegreen', zorder=4, s=1, alpha=.5)
        
    plt.scatter(X[:,0], X[:,1], c=y, zorder=3, s=10)
    plt.xlim(xlim), plt.ylim(ylim)
    plt.xlabel(r'$x_1$'), plt.ylabel(r'$x_2$')
    plt.gca().set_aspect('equal')


def select_bandwidth(X, q, device='cpu'):
    D = pairwise_distance(X, device=device)
    D = D.flatten()
    D.sort()
    D = np.trim_zeros(D)
    idx = np.floor(q*len(D)).astype(np.int)
    return (D[idx]/2)**.5


def images(X, M=5, N=20, save=None):
    X = X.reshape([M,N,256,256,3])
    X = np.pad(X,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(1,1))
    X = X.transpose([0,2,1,3,4]).reshape([M*260,N*260,3])
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off'); plt.xticks([]); plt.yticks([])
    plt.imshow(X)
    if save is not None:
        plt.savefig(save)


def pairwise_distance(X, Y=None, device='cpu'):
    if Y is None:
        return pdist(X, metric='sqeuclidean')
    else:
        return cdist(X, Y, metric='sqeuclidean')


def plot_kernel_matrix(K):
    absmax = abs(K).max()
    plt.title(r'Kernel matrix')
    plt.imshow(K, vmin=-absmax, vmax=absmax, cmap='seismic')
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.colorbar()


def plot_kernel_eigvals(K):
    U, V = np.linalg.eigh(K)
    U = U[::-1]
    absmax = abs(U).max()
    plt.title(r'Eigenvalues of the kernel matrix')
    plt.bar(range(len(U)), U, width=1.0)
    plt.xlabel(r'$i=1,\ldots,N$')
    plt.ylabel(r'$\lambda_i$')
    plt.ylim([-1.2*absmax, 1.2*absmax])


def plot_kernel(K):
    plt.figure(figsize=(8,3))
    plt.subplot(121)
    plot_kernel_matrix(K)
    plt.axis('equal')
    plt.subplot(122)
    plot_kernel_eigvals(K)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()


def PCA(Xtrain, Xtest, n_components=2):
    pca = sklPCA(n_components)
    Xtrain = pca.fit_transform(Xtrain)
    Xtest = pca.transform(Xtest)
    return Xtrain, Xtest
