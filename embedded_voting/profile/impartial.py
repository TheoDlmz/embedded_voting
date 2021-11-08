# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.profile.ratings import Ratings


class ImpartialCulture(Ratings):
    """
    Generate random ratings based on impartial culture (voters are independent)
    """
    def __init__(self, n_voters, n_candidates):
        super().__init__(np.random.rand(n_voters, n_candidates))
