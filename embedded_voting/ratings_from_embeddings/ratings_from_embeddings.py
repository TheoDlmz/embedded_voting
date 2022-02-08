# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""


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


