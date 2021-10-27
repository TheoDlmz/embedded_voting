# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings


class Embedder:
    def __init__(self, n_dim=0):
        self.n_dim = n_dim

    def __call__(self, scores):
        raise NotImplementedError


class RandomEmbedder(Embedder):
    def __call__(self, scores):
        n_voters = scores.shape[0]
        embs = np.abs(np.random.randn(n_voters, self.n_dim))
        return Embeddings(embs).normalize()











