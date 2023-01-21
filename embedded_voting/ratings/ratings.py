# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import numpy as np


# noinspection PyUnresolvedReferences
class Ratings(np.ndarray):
    """
    Ratings of the voters in a given election.

    Parameters
    ----------
    ratings: list, np.ndarray or Ratings
        The ratings given by each voter to each candidate.

    Attributes
    ----------
    n_voters: int
        The number of voters.

    n_candidates: int
        The number of candidates.

    Examples
    --------
    >>> ratings = Ratings([[1, .8, .5], [.3, .5, .9]])
    >>> ratings
    Ratings([[1. , 0.8, 0.5],
             [0.3, 0.5, 0.9]])
    >>> ratings.n_voters
    2
    >>> ratings.n_candidates
    3
    >>> ratings.voter_ratings(0)
    array([1. , 0.8, 0.5])
    >>> ratings.candidate_ratings(0)
    array([1. , 0.3])

    """
    def __new__(cls, ratings):
        obj = np.asarray(ratings).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        if len(self.shape) == 2:
            self.n_voters, self.n_candidates = self.shape

    def voter_ratings(self, i):
        return np.array(self[i:i+1, :])[0]

    def candidate_ratings(self, i):
        return np.array(self[:, i:i+1]).T[0]
