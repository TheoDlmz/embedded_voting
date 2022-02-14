# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

from embedded_voting.rules.singlewinner_rules.rule_fast_nash import RuleFastNash
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
import numpy as np


class Aggregator:
    """
    A class for an election generator with memory.
    You can run an election by calling
    it with the matrix of scores.

    Parameters
    ----------
    rule: Rule
        The aggregation rule you want to use in your elections. Default is :class:`~embedded_voting.RuleFastNash`

    Attributes
    ----------
    embeddings: Embeddings
        The current embeddings of the voters

    ratings_history: np.ndarray
        The history of all ratings given by the voter. Is used to compute correlations between
        voters.

    rule: Rule
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

    def __init__(self, rule=None, embeddings_from_ratings=None, default_train=False, name="aggregator"):
        if rule is None:
            rule = RuleFastNash()
        self.rule = rule
        self.embeddings = None
        self.ratings_history = None
        if embeddings_from_ratings is None:
            self.embeddings_from_ratings = EmbeddingsFromRatingsCorrelation()
        else:
            self.embeddings_from_ratings = embeddings_from_ratings
        self.default_train = default_train
        self.name = name

    def __call__(self, ratings, train=None):
        """
        This function run an election using the :attr:`embeddings_from_ratings` and the scores.

        Parameters
        ----------
        ratings: np.ndarray or list
            The matrix of scores given by the voters. ``ratings[i,j]`` corresponds to the
            score given by the voter i to candidate j.

        train: bool
            If True, we retrain the :attr:`embeddings_from_ratings` before doing the election (using the
            data of the election).
        """
        ratings = Ratings(ratings)
        if self.ratings_history is None:
            self.ratings_history = ratings
        else:
            self.ratings_history = np.concatenate([self.ratings_history, ratings], axis=1)

        if self.embeddings is None or (train is None and self.default_train) or train:
            self.train()

        self.rule.delete_cache()

        return self.rule(ratings, self.embeddings)

    def train(self):
        """
        This function can be used to train the embeddings_from_ratings on the newest data
        it gathered during the recent elections.

        Return
        ------
        Aggregator
            The object
        """
        self.embeddings = self.embeddings_from_ratings(self.ratings_history)
        return self

    def reset(self):
        """
        This function reset the embeddings and ratings history of the aggregator
        """
        self.embeddings = None
        self.ratings_history = None
