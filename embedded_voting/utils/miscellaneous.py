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
