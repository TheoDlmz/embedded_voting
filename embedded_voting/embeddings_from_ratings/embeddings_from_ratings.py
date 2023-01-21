# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings


class EmbeddingsFromRatings:
    """
    An abstract class that convert ratings into embeddings using some function.
    """

    def __call__(self, ratings):
        """
        Compute the embeddings.

        Parameters
        ----------
        ratings: Ratings or np.ndarray
            Ratings given by the voters to the candidates

        Return
        ------
        embeddings: Embeddings
        """
        raise NotImplementedError
