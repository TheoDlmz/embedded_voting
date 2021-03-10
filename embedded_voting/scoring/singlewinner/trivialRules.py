# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

from embedded_voting.scoring.singlewinner.general import ScoringRule


class SumScores(ScoringRule):
    """
    Voting rule that rank candidates by the sum of their scores

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """
    def score_(self, candidate):
        return self.profile_.scores[::, candidate].sum()


class ProductScores(ScoringRule):
    """
    Voting rule that rank candidates by the products of their scores

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election


    """
    def __init__(self, profile):
        super().__init__(profile)
        self.score_components = 2

    def score_(self, candidate):
        scores = self.profile_.scores[::, candidate]
        count = 0
        prod = 1
        for s in scores:
            if s > 0:
                count += 1
                prod *= s
        return count, prod
