# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np


def normalize(x):
    """
    Normalize the input vector.

    Parameters
    ----------
    x : np.ndarray or list

    Return
    ------
    np.ndarray
        `x` divided by its Euclidean norm.

    Examples
    --------
    >>> my_vector = np.arange(3)
    >>> normalize(my_vector)
    array([0.        , 0.4472136 , 0.89442719])

    >>> my_vector = [0, 1, 2]
    >>> normalize(my_vector)
    array([0.        , 0.4472136 , 0.89442719])
    """
    return x / np.linalg.norm(x)


def max_angular_dilatation_factor(vector, center):
    """
    Maximum angular dilatation factor to stay in the positive orthant.

    Consider `center` and `vector` two unit vectors of the positive orthant. Consider a "spherical dilatation" that
    moves `vector` by multiplying the angle between `center` and `vector` by a given dilatation factor. The question is:
    what is the maximal value of this dilatation factor so that the result still is in the positive orthant?

    More formally, there exists a unit vector `unit_orthogonal` and an angle `theta in [0, pi/2]` such that
    `vector = cos(theta) * center + sin(theta) * unit_orthogonal`. Then there exists a maximal angle
    `theta_max in [0, pi/2]` such that `cos(theta_max) * center + sin(theta_max) * unit_orthogonal`
    is still in the positive orthant. We define the maximal angular dilatation factor as `theta_max / theta`.

    Parameters
    ----------
    vector : np.ndarray
        A unit vector in the positive orthant.
    center : np.ndarray
        A unit vector in the positive orthant.

    Returns
    -------
    float
        The maximal angular dilatation factor. If `vector` is equal to `center`, then `np.inf` is returned.

    Examples
    --------
    >>> max_angular_dilatation_factor(
    ...     vector=np.array([1, 0]),
    ...     center=np.array([1, 1]) * np.sqrt(1/2)
    ... )
    1.0
    >>> max_angular_dilatation_factor(
    ...     vector=np.array([1, 1]) * np.sqrt(1/2),
    ...     center=np.array([1, 0])
    ... )
    2.0

    >>> my_center = np.array([1., 1., 1.]) * np.sqrt(1/3)
    >>> my_unit_orthogonal = np.array([1, -1, 0]) * np.sqrt(1/2)
    >>> my_theta = np.pi / 9
    >>> my_vector = np.cos(my_theta) * my_center + np.sin(my_theta) * my_unit_orthogonal
    >>> k = max_angular_dilatation_factor(vector=my_vector, center=my_center)
    >>> k
    1.9615760241796105
    >>> dilated_vector = np.cos(k * my_theta) * my_center + np.sin(k * my_theta) * my_unit_orthogonal
    >>> np.round(dilated_vector, 4)
    array([0.8944, 0.    , 0.4472])
    """
    scalar_product = vector @ center
    vector_collinear = scalar_product * center  # Component of `vector` on the direction of `center`.
    vector_orthogonal = vector - vector_collinear  # Component of `vector` in the orthogonal direction.
    norm_vector_orthogonal = np.linalg.norm(vector_orthogonal)
    if norm_vector_orthogonal == 0:
        return np.inf
    unit_orthogonal = vector_orthogonal / np.linalg.norm(vector_orthogonal)
    mask = unit_orthogonal < 0
    if not any(mask):
        theta_max = np.pi / 2
    else:
        theta_max = np.arctan(np.min(- center[mask] / unit_orthogonal[mask]))
    theta = np.arccos(scalar_product)
    return theta_max / theta
