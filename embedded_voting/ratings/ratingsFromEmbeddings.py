# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings


class RatingsFromEmbeddings:
    """
    This abstract class is used to generate ratings from embeddings

    Parameters
    ----------
    n_candidates: int
        The number of candidates wanted in the ratings

    Attributes
    ----------
    n_candidates: int
        The number of candidates wanted in the ratings
    """

    def __init__(self, n_candidates):
        self.n_candidates = n_candidates

    def __call__(self, embeddings, *args):
        """
        This method generate ratings from the embeddings
        """
        raise NotImplementedError


class RatingsFromEmbeddingsCorrelated(RatingsFromEmbeddings):
    """
    This method create ratings correlated to the embeddings by a score matrix

    Parameters
    ----------
    n_candidates: int
        The number of candidates wanted in the ratings
    n_dim: int
        The number of dimension of the embeddings
    scores_matrix: np.ndarray or list
        An array with shape ``n_dim, n_candidates`` such that ``scores_matrix[i,j]`` determines the rating
        given by the group of voter in the dimension i to candidate j. If none is specified, a random one
        is generated

    Attributes
    ----------
    n_candidates: int
        The number of candidates wanted in the ratings
    n_dim: int
        The number of dimension of the embeddings
    scores_matrix: np.ndarray or list
        An array with shape ``n_dim, n_candidates`` such that ``scores_matrix[i,j]`` determines the rating
        given by the group of voter in the dimension i to candidate j. If

    Examples
    --------
    >>> np.random.seed(42)
    >>> embeddings = Embeddings(np.array([[0, 1], [1, 0], [1, 1]]), norm=True)
    >>> generator = RatingsFromEmbeddingsCorrelated(2, 3, scores_matrix=np.array([[.8,.4],[.1,.7]]))
    >>> generator(embeddings, .5)
    Ratings([[0.23727006, 0.82535715],
             [0.76599697, 0.49932924],
             [0.30300932, 0.35299726]])
    """

    def __init__(self, n_candidates, n_dim, scores_matrix=None):
        super().__init__(n_candidates)
        self.n_dim = n_dim
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.scores_matrix = np.array(scores_matrix)

    def __call__(self, embeddings, coherence=0, *args):
        """
        This method generate ratings from the embeddings using the score matrix. The coherence parameters
        indicate how much we should take into account the score matrix.

        Parameters
        ----------
        embeddings: Embeddings
            The embeddings we want to use to obtain the ratings

        coherence: float
            Between 0 and 1, indicates the desired level of correlation between embeddings and ratings. If 0,
            ratings are random, if 1, ratings are perfectly correlated to embeddings.

        Return
        ------
        Ratings
        """
        embeddings = Embeddings(embeddings)
        positions = np.array(embeddings)
        n_voters = embeddings.n_voters
        ratings = coherence * (positions ** 2).dot(self.scores_matrix)
        ratings += (1 - coherence) * np.random.rand(n_voters, self.n_candidates)
        ratings = np.minimum(ratings, 1)
        ratings = np.maximum(ratings, 0)
        return Ratings(ratings)

    def set_scores(self, scores_matrix=None):
        """
        Update the scores matrix of
        the parametric ratings.

        Parameters
        ----------
        scores_matrix : np.ndarray or list
            Matrix of shape :attr:`n_dim`, :attr:`n_candidates` containing the scores given by
            each group. More precisely, `scores_matrix[i,j]` is the score given by the group
            represented by the dimension i to the candidate j.
            By default, it is set at random with a uniform distribution.

        Return
        ------
        ProfileGenerator
            The object itself.

        Examples
        --------
        >>> generator = RatingsFromEmbeddingsCorrelated(2, 3)
        >>> generator.set_scores(np.array([[.8,.4],[.1,.7]]))
        <embedded_voting.ratings.ratingsFromEmbeddings.RatingsFromEmbeddingsCorrelated object at ...>
        >>> generator.scores_matrix
        array([[0.8, 0.4],
               [0.1, 0.7]])
        """
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.scores_matrix = np.array(scores_matrix)
        return self
