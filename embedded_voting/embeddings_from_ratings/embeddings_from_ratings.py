# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np

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
