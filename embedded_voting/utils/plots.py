# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import ternary
import numpy as np


def create_3d_plot(fig, position=None):
    """
    Create the background for a 3D plot on the non-negative orthant.

    Parameters
    ----------
    fig
        The matplotlib figure on which we are drawing.
    position
        The position of the subplot on which we are drawing.

    Return
    ------
    matplotlib ax
    """
    if position is None:
        position = [1, 1, 1]

    ax = fig.add_subplot(position[0], position[1], position[2], projection='3d')

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_zlim(0, 1)

    angle = np.linspace(0, np.pi / 2, 100)
    n_angles = len(angle)
    cos_x = [np.cos(a) for a in angle]
    sin_x = [np.sin(a) for a in angle]

    ax.plot(cos_x, sin_x, [0] * n_angles, color='k', alpha=0.5)
    ax.plot([0] * n_angles, cos_x, sin_x, color='k', alpha=0.5)
    ax.plot(cos_x, [0] * n_angles, sin_x, color='k', alpha=0.5)

    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax


def create_ternary_plot(fig, position=None):
    """
    Create the background for a 2D ternary plot of the non-negative orthant.

    Return
    ______
    matplotlib ax
    """
    if position is None:
        position = [1, 1, 1]
    ax = fig.add_subplot(position[0], position[1], position[2])
    figure, tax = ternary.figure(ax=ax, scale=1)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.1, color="blue")
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, ticks=["%.1f" % (i/10) for i in range(11)])
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    tax.left_axis_label("$x$", fontsize=14)
    tax.right_axis_label("$y$", fontsize=14)
    tax.bottom_axis_label("$z$", fontsize=14)
    return tax


def create_map_plot(fig, image, position, title=""):
    """
    Create the background for a map plot.

    Parameters
    ----------
    fig : matplotlib figure
        The matplotlib figure on which we are drawing.
    image : np.ndarray
        The image to plot. Should be of size `map_size`, `map_size`.
    position: list
        The position of the subplot on which we are drawing.
    title : str
        Title of the plot.

    Return
    ------
    matplotlib ax
    """
    map_size = len(image)
    # noinspection PyUnresolvedReferences
    ax = fig.add_subplot(position[0], position[1], position[2])
    ax.imshow(image[::-1, ::], vmin=0, vmax=1)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Orthogonality')
    ax.set_title(title)
    ax.set_xticks([0, map_size - 1])
    ax.set_yticks([0, map_size - 1])
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([1, 0])

    return ax
