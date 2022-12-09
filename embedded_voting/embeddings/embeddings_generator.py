# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
from embedded_voting.embeddings.embeddings import Embeddings


class EmbeddingsGenerator:
    """
    This abstract class creates Embeddings from scratch using some function.

    Parameters
    __________
    n_voters: int
        Number of voters in the embeddings.
    n_dim: int
        Number of dimensions for the embeddings.
    """
    def __init__(self, n_voters, n_dim):
        self.n_voters = n_voters
        self.n_dim = n_dim

    def __call__(self, *args):
        """
        This function creates embeddings.

        Return
        ------
        Embeddings
        """
        raise NotImplementedError
