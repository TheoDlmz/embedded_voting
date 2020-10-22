import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize(x):
    """
    Normalize the input vector

    Parameters
    ______
    x : np.array

    Return
    ______
    np.array
        The normalized vector of x

    """
    return x / np.sqrt(np.sum([x_i ** 2 for x_i in x]))



def create_3D_plot(fig, intfig=[1, 1, 1]):
    ax = fig.add_subplot(intfig[0], intfig[1], intfig[2], projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_zlim(0, 1)

    angle = np.linspace(0, np.pi / 2, 100)
    n_angles = len(angle)
    cosx = [np.cos(a) for a in angle]
    sinx = [np.sin(a) for a in angle]
    ax.plot(cosx, sinx, [0] * n_angles, color='k', alpha=0.5)
    ax.plot([0] * n_angles, cosx, sinx, color='k', alpha=0.5)
    ax.plot(cosx, [0] * n_angles, sinx, color='k', alpha=0.5)

    # plt.axis('off')
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax
