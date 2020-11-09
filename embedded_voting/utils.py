# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""


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


def create_2D_plot(fig, intfig=[1, 1, 1]):
    ax = fig.add_subplot(intfig[0], intfig[1], intfig[2])
    ax.set_xlim(np.pi/2+0.15, 0-0.15)
    ax.set_ylim(np.pi/2+0.15, 0-0.15)

    ax.grid(False)
    ax.set_ylabel("Phi")
    ax.set_xlabel("Theta")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def _cache(f):
    """
    Auxiliary decorator used by ``cached_property``.

    :param f: a method with no argument (except ``self``).
    :return: the same function, but with a `caching' behavior.
    """
    name = f.__name__

    # noinspection PyProtectedMember
    def _f(*args):
        try:
            return args[0]._cached_properties[name]
        except KeyError:
            # Not stored in cache
            value = f(*args)
            args[0]._cached_properties[name] = value
            return value
        except AttributeError:
            # cache does not even exist
            value = f(*args)
            args[0]._cached_properties = {name: value}
            return value
    _f.__doc__ = f.__doc__
    return _f


def cached_property(f):
    """
    Decorator used in replacement of @property to put the value in cache automatically.

    The first time the attribute is used, it is computed on-demand and put in cache. Later accesses to the
    attributes will use the cached value.

    Cf. :class:`DeleteCacheMixin` for an example.
    """
    return property(_cache(f))


class DeleteCacheMixin:
    """
    Mixin used to delete cached properties.

    Cf. decorator :meth:`cached_property`.

    >>> class Example(DeleteCacheMixin):
    ...     @cached_property
    ...     def x(self):
    ...         print('Big computation...')
    ...         return 6 * 7
    >>> a = Example()
    >>> a.x
    Big computation...
    42
    >>> a.x
    42
    >>> a.delete_cache()
    >>> a.x
    Big computation...
    42
    """

    def delete_cache(self) -> None:
        self._cached_properties = dict()
