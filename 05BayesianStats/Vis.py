import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal as gaussian


def plot_gaussian(mu, Sigma, N=100, z_offset=-0.2, val=2):
    gauss = gaussian(mu, Sigma)
    x = np.linspace(mu[0]-val, mu[0]+val, N)
    y = np.linspace(mu[1]-val, mu[1]+val, N)
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    z = gauss.pdf(pos)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.rainbow)

    ax.contourf(x, y, z, zdir='z', offset=z_offset, cmap=cm.rainbow)

    max_z = np.max(z)
    ax.set_zlim(z_offset, max_z+0.2*max_z)
    ax.set_zticks(np.linspace(0, max_z+0.2*max_z, 5))
    ax.view_init(15, -21)

    plt.show()


if __name__ == '__main__':

    mu = np.zeros(2)
    Sigma = 0.5*np.eye(2)

    plot_gaussian(mu, Sigma, val=1)