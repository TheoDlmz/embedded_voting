# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.profile import Profile
from embedded_voting.embeddings.embeddings import Embeddings


class SumScores(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the scores given by
    the voters.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.

    Attributes
    ----------
    profile : Profile
        The profile of voters on which we run the election.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> embeddings = np.array([[1, 1], [1, 0], [0, 1]])
    >>> profile = Profile(scores, Embeddings(embeddings))
    >>> election = SumScores(profile)
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
        return self.profile_.ratings[::, candidate].sum()


class ProductScores(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the product of the scores given by
    the voters.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.

    Attributes
    ----------
    profile : Profile
        The profile of voters on which we run the election.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> embeddings = np.array([[1, 1], [1, 0], [0, 1]])
    >>> profile = Profile(scores, Embeddings(embeddings))
    >>> election = ProductScores(profile)
    >>> election.scores_
    [(3, 0.06999999999999999), (2, 0.6), (3, 0.048)]
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.6857142857142858]
    """

    def __init__(self, profile=None):
        super().__init__(profile)
        self._score_components = 2

    def _score_(self, candidate):
        scores = self.profile_.ratings[::, candidate]
        count = 0
        prod = 1
        for s in scores:
            if s > 0:
                count += 1
                prod *= s
        return count, prod
