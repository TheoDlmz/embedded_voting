# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings.embedder import IdentityEmbedder


class Ratings:
    """
    This class is a simple class that contains information about the ratings of voters in a given election

    Parameters
    ----------
    ratings: np.ndarray
        The ratings given by each voter to each candidate

    Attributes
    ----------
    ratings: np.ndarray
        The ratings given by each voter to each candidate

    n_voters: int
        The number of voters

    n_candidates: int
        The number of candidates

    Examples
    --------
    >>> profile = Ratings(np.array([[1, .8, .5], [.3, .5, .9]]))
    >>> profile.ratings
    array([[1. , 0.8, 0.5],
           [0.3, 0.5, 0.9]])
    >>> profile.n_voters
    2
    >>> profile.n_candidates
    3

    """
    def __init__(self, ratings):
        self.ratings = ratings
        self.n_voters, self.n_candidates = ratings.shape

    def generate_embeddings(self, embedder):
        """
        A function to generate embeddings from the ratings using an embedder

        Parameters
        ----------
        embedder: Embedder
            The embedder used to generates embeddings

        Return
        ------
        Embeddings
            The generated embeddings

        Examples
        --------
        >>> profile = Ratings(np.array([[1, .8, .5], [.3, .5, .9]]))
        >>> embs = profile.generate_embeddings(IdentityEmbedder())
        >>> embs.positions
        array([[1., 0.],
               [0., 1.]])

        """
        return embedder(self.ratings)
