# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile


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
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SumScores(my_profile)
    >>> election.scores_
    [1.4, 1.6, 1.3]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.3333333333333328, 1.0, 0.0]
    """
    def score_(self, candidate):
        return self.profile_.scores[::, candidate].sum()


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
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = ProductScores(my_profile)
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

    def score_(self, candidate):
        scores = self.profile_.scores[::, candidate]
        count = 0
        prod = 1
        for s in scores:
            if s > 0:
                count += 1
                prod *= s
        return count, prod
