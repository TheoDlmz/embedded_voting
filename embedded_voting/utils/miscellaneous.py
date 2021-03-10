# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np


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
    return x / np.linalg.norm(x)
