# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
from embedded_voting.ratings.ratings import Ratings
import numpy as np

class RatingsGenerator:
    """
    This abstract class creates :class:`Ratings` from scratch using some function.

    Parameters
    __________
    n_voters: int
        Number of voters in the embeddings.
    """
    def __init__(self, n_voters):
        self.n_voters = n_voters

    def __call__(self, n_candidates):
        """
        This function creates ratings.

        Parameters
        ----------
        n_candidates : int
            The number of candidates.

        Return
        ------
        Ratings
        """
        raise NotImplementedError

    def save_scores(self, n_candidates, filename):
        ratings = self(n_candidates)
        ratings.tofile('%s.csv' % filename, sep=',')


