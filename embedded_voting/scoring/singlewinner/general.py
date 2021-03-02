# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property


class ScoringFunction(DeleteCacheMixin):
    """
    The general class of scoring functions

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election


    """

    def __init__(self, profile=None):
        self.profile_ = None
        if profile is not None:
            self(profile)

    def __call__(self, profile):
        self.profile_ = profile
        self.delete_cache()
        return self

    def score_(self, cand):
        """
        Compute the score of some candidate

        Parameters
        _____
        cand : int
            Index of the candidate for which we want the score
        """
        raise NotImplementedError

    @cached_property
    def scores_(self):
        """
        Compute the score of all candidates
        """
        return [self.score_(cand) for cand in range(self.profile_.m)]

    @cached_property
    def ranking_(self):
        """
        Compute the ranking over all candidates
        """
        return np.argsort(self.scores_)[::-1]

    @cached_property
    def winner_(self):
        """
        Compute the winner of the election
        """
        return self.ranking_[0]

    @cached_property
    def welfare_(self):
        """
        Compute the score of all candidates
        """
        scores = self.scores_
        max_score = np.max(scores)
        min_score = np.min(scores)
        if max_score == min_score:
            return np.ones(self.profile_.m)
        return (scores - min_score)/(max_score - min_score)

    def plot_winner(self):
        """
        Plot the winner of the election on a 3D Plot
        """
        raise NotImplementedError


