# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""

from embedded_voting.rules.singlewinner_rules.rule_fast_nash import RuleFastNash
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.utils.miscellaneous import center_and_normalize
import numpy as np


class Aggregator:
    """
    A class for an election generator with memory.

    You can run an election by calling it with the matrix of ratings.

    Parameters
    ----------
    rule: Rule
        The aggregation rule you want to use in your elections. Default is :class:`~embedded_voting.RuleFastNash`
    embeddings_from_ratings: EmbeddingsFromRatings
        If no embeddings are specified in the call, this `EmbeddingsFromRatings` object is use to generate
        the embeddings from the ratings. Default:
        `EmbeddingsFromRatingsCorrelation(preprocess_ratings=center_and_normalize)`.
    default_train: bool
        If True, then by default, train the embeddings at each election.
    name: str, optional
        Name of the aggregator.
    default_add: bool
        If True, then by default, add the ratings to the history.

    Attributes
    ----------
    ratings_history: np.ndarray
        The history of all ratings given by the voters.
    embeddings: Embeddings
        The current embeddings of the voters.

    Examples
    --------
    >>> aggregator = Aggregator()
    >>> results = aggregator([[7, 5, 9, 5, 1, 8], [7, 5, 9, 5, 2, 7], [6, 4, 2, 4, 4, 6], [3, 8, 1, 3, 7, 8]])
    >>> results.embeddings_
    Embeddings([[ 1.        ,  0.98602958,  0.01549503, -0.43839669],
                [ 0.98602958,  1.        , -0.09219821, -0.54916602],
                [ 0.01549503, -0.09219821,  1.        ,  0.43796787],
                [-0.43839669, -0.54916602,  0.43796787,  1.        ]])
    >>> results.ranking_
    [5, 0, 1, 3, 4, 2]
    >>> results.winner_
    5
    >>> results = aggregator([[2, 4, 8], [9, 2, 1], [0, 2, 5], [4, 5, 3]])
    >>> results.ranking_
    [2, 1, 0]
    """

    def __init__(self, rule=None, embeddings_from_ratings=None, default_train=True, name="aggregator", default_add=True):
        if rule is None:
            rule = RuleFastNash()
        if embeddings_from_ratings is None:
            embeddings_from_ratings = EmbeddingsFromRatingsCorrelation(preprocess_ratings=center_and_normalize)
        # Parameters
        self.rule = rule
        self.embeddings_from_ratings = embeddings_from_ratings
        self.default_train = default_train
        self.default_add = default_add
        self.name = name
        # Variables
        self.ratings_history = None
        self.embeddings = None

    def __call__(self, ratings, train=None):
        """
        Run an election.

        Parameters
        ----------
        ratings: Ratings or np.ndarray or list
            The matrix of ratings given by the voters.
        train: bool
            Determine the policy for training or not (cf. below).

        Notes
        -----
        * If `embeddings` is None (never computed before, or just reset), then train, whatever the parameters.
        * If `train` is True, then train.
        * If `train` is None and if `default_train` is True, then train.
        """
        ratings = Ratings(ratings)
        if self.ratings_history is None:
            self.ratings_history = ratings
        elif self.default_add:
            self.ratings_history = np.concatenate([self.ratings_history, ratings], axis=1)
        if self.embeddings is None or train or (train is None and self.default_train):
            self.train()
        return self.rule(ratings, self.embeddings)

    def train(self):
        """
        Update the variable `embeddings`, based on `ratings_history`.

        Return
        ------
        Aggregator
            The object itself.
        """
        self.embeddings = self.embeddings_from_ratings(self.ratings_history)
        return self

    def reset(self):
        """
        Reset the variables `ratings_history` and `embeddings`.

        Return
        ------
        Aggregator
            The object itself.
        """
        self.ratings_history = None
        self.embeddings = None
        return self
