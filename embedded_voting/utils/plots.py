# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import ternary
import numpy as np

def create_3D_plot(fig, intfig=None):
    """
    Create the background for a 3D plot on the positive orthan.

    Parameters
    ______
    fig : The matplotlib figure on which we are drawing
    intfig : The intfig of the subplot on which we are drawing

    Return
    ______
    matplotlib.pyplot.ax

    """
    if intfig is None:
        intfig = [1, 1, 1]
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

    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax


def create_ternary_plot(fig, intfig=None):
    """
        Create the background for a 2D ternary plot of the positive orthan.

        Return
        ______
        matplotlib.pyplot.ax

    """
    if intfig is None:
        intfig = [1, 1, 1]
    ax = fig.add_subplot(intfig[0], intfig[1], intfig[2], projection='3d')
    figure, tax = ternary.figure(ax=ax, scale=1)
    figure.set_size_inches(5, 5)

    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.1, color="blue")
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, ticks=["%.1f"%(i/10) for i in range(11)])
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    tax.left_axis_label("$x$", fontsize=14)
    tax.right_axis_label("$y$", fontsize=14)
    tax.bottom_axis_label("$z$", fontsize=14)
    return tax
