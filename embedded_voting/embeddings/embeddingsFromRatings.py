# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np

from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


class EmbeddingsFromRatings:
    """
    An abstract class that convert ratings into embeddings using some function

    """

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


class EmbeddingsFromRatingsRandom(EmbeddingsFromRatings):
    """
    Generates random normalized embeddings for the voters

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsRandom(2)
    >>> generator(ratings)
    Embeddings([[0.96337365, 0.26816265],
                [0.39134578, 0.92024371],
                [0.70713157, 0.70708199],
                [0.89942118, 0.43708299],
                [0.65433791, 0.75620229]])
    """
    def __init__(self, n_dim=0):
        self.n_dim = n_dim

    def __call__(self, ratings):
        ratings = Ratings(ratings)
        n_voters = ratings.shape[0]
        embs = np.abs(np.random.randn(n_voters, self.n_dim))
        return Embeddings(embs, norm=True)


class EmbeddingsFromRatingsIdentity(EmbeddingsFromRatings):
    """
    Use the identity matrix as the embeddings for the voters

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsIdentity()
    >>> generator(ratings)
    Embeddings([[1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.]])
    """
    def __call__(self, ratings):
        ratings = Ratings(ratings)
        n_dim = ratings.shape[0]
        return Embeddings(np.eye(n_dim))


class EmbeddingsFromRatingsSelf(EmbeddingsFromRatings):
    """
    Use the normalized ratings as the embeddings for the voters

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsSelf()
    >>> generator(ratings)
    Embeddings([[0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027]])
    """
    def __call__(self, ratings):
        return Embeddings(ratings, norm=True)


class EmbeddingsFromRatingsCorrelation(EmbeddingsFromRatings):
    """
    Use the correlation with each voter as the embeddings

    Attributes
    ----------
    n_sing_val_: int
        The number of relevant singular values when we compute the SVD. based on the Principal Component
        Analysis (PCA)

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsCorrelation()
    >>> generator(ratings)
    Embeddings([[0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136]])
    >>> generator.n_sing_val_
    1

    """
    def __init__(self):
        super().__init__()
        self.n_sing_val_ = None

    def __call__(self, ratings):
        ratings = Ratings(ratings)
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

        self.n_sing_val_ = n_v
        return Embeddings(np.dot(positions, positions.T))
