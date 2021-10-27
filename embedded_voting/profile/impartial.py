# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.profile.profile import Profile


class ImpartialCulture(Profile):
    def __init__(self, n_voters, n_candidates, embeddings=None):
        super().__init__(np.random.rand(n_voters, n_candidates), embeddings=embeddings)
