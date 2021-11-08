# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings


class Embedder:
    """
    An abstract class that convert ratings into embeddings using some function

    Parameters
    ----------
    n_dim: int
        The number of dimension wanted for the embeddings. If 0, the embedder chose the number of
        dimension by itself

    Attributes
    ----------
    n_dim: int
        The number of dimension wanted for the embeddings. If 0, the embedder chose the number of
        dimension by itself
    """
    def __init__(self, n_dim=0):
        self.n_dim = n_dim

    def __call__(self, ratings):
        """
        This function takes as input the ratings and return the embeddings

        Parameters
        ----------
        ratings: np.ndarray
            Ratings given by the voters to the candidates

        Return
        ------
        Embeddings
        """
        raise NotImplementedError


class RandomEmbedder(Embedder):
    """
    Generates random normalized embeddings for the voters
    """
    def __call__(self, ratings):
        n_voters = ratings.shape[0]
        embs = np.abs(np.random.randn(n_voters, self.n_dim))
        return Embeddings(embs, norm=True)


class IdentityEmbedder(Embedder):
    """
    Use the identity matrix as the embeddings for the voters
    """
    def __call__(self, ratings):
        self.n_dim = ratings.shape[0]
        return Embeddings(np.eye(self.n_dim))


class SelfEmbedder(Embedder):
    """
    Use the normalized ratings as the embeddings for the voters
    """
    def __call__(self, ratings):
        return Embeddings(ratings, norm=True)


class CorrelationEmbedder(Embedder):
    """
    Use the correlation with each voter as the embeddings

    Attributes
    ----------
    n_sing_val: int
        The number of relevant singular values when we compute the SVD. based on the Principal Component
        Analysis (PCA)

    """
    def __init__(self):
        super().__init__()
        self.n_sing_val = 0

    def __call__(self, ratings):
        positions = (ratings.T / np.sqrt((ratings ** 2).sum(axis=1))).T
        n_voters, n_candidates = ratings.shape
        self.n_dim = n_candidates

        u, s, v = np.linalg.svd(positions)

        n_voters, n_candidates = positions.shape
        s = np.sqrt(s)
        s /= s.sum()
        n_v = 0
        for s_e in s:
            if s_e >= max(1 / n_voters, 1 / n_candidates):
                n_v += 1

        self.n_sing_val = n_v
        return Embeddings(np.dot(positions, positions.T))
