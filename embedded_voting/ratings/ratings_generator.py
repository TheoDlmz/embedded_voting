# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
from embedded_voting.ratings.ratings import Ratings


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
