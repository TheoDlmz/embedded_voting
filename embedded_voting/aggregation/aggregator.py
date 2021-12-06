# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

from embedded_voting.scoring.singlewinner.fast import FastNash
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddingsFromRatings import EmbeddingsFromRatingsCorrelation
import numpy as np


class Aggregator:
    """
    A class for an election generator with memory.
    You can run an election by calling
    it with the matrix of scores.

    Parameters
    ----------
    rule: ScoringRule
        The aggregation rule you want to use in your elections. Default is :class:`~embedded_voting.FastNash`

    Attributes
    ----------
    embeddings: Embeddings
        The current embeddings of the voters

    ratings_history: np.ndarray
        The history of all ratings given by the voter. Is used to compute correlations between
        voters.

    rule: ScoringRule
        The scoring rule used for the elections.

    Examples
    --------
    >>> aggregator = Aggregator()
    >>> results = aggregator([[7, 5, 9, 5, 1, 8], [7, 5, 9, 5, 2, 7], [6, 4, 2, 4, 4, 6], [3, 8, 1, 3, 7, 8]])
    >>> results.ranking_
    [5, 1, 0, 3, 4, 2]
    >>> results.winner_
    5
    >>> results = aggregator([[2, 4, 8], [9, 2, 1], [0, 2, 5], [4, 5, 3]], train=True)
    >>> results.ranking_
    [2, 0, 1]
    """

    def __init__(self, rule=None):
        if rule is None:
            rule = FastNash()
        self.rule = rule
        self.embeddings = None
        self.ratings_history = None

    def __call__(self, ratings, train=False):
        """
        This function run an election using the :attr:`embedder` and the scores.

        Parameters
        ----------
        ratings: np.ndarray or list
            The matrix of scores given by the voters. ``ratings[i,j]`` corresponds to the
            score given by the voter i to candidate j.

        train: bool
            If True, we retrain the :attr:`embedder` before doing the election (using the
            data of the election).
        """
        ratings = Ratings(ratings)
        if self.ratings_history is None:
            self.ratings_history = ratings
        else:
            self.ratings_history = np.concatenate([self.ratings_history, ratings], axis=1)

        if self.embeddings is None or train:
            self.train()

        self.rule.delete_cache()

        return self.rule(ratings, self.embeddings)

    def train(self):
        """
        This function can be used to train the embedder on the newest data
        it gathered during the recent elections.

        Return
        ------
        Aggregator
            The object
        """
        embedder = EmbeddingsFromRatingsCorrelation()
        self.embeddings = embedder(self.ratings_history)
        self.rule.n_v = embedder.n_sing_val_
        return self
