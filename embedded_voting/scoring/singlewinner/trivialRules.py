# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings


class SumScores(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the scores given by
    the voters.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]))
    >>> election = SumScores()(ratings, embeddings)
    >>> election.scores_
    [1.4, 1.6, 1.3]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.3333333333333328, 1.0, 0.0]
    """
    def _score_(self, candidate):
        return self.ratings_.candidate_ratings(candidate).sum()


class ProductScores(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the product of the scores given by
    the voters.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]))
    >>> election = ProductScores()(ratings, embeddings)
    >>> election.scores_
    [(3, 0.06999999999999999), (2, 0.6), (3, 0.048)]
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.6857142857142858]
    """

    def __init__(self):
        super().__init__(score_components=2)

    def _score_(self, candidate):
        scores = self.ratings_.candidate_ratings(candidate)
        count = 0
        prod = 1
        for s in scores:
            if s > 0:
                count += 1
                prod *= s
        return count, prod
