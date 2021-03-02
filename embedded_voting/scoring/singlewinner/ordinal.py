# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringFunction


class PositionalRuleExtension(ScoringFunction):
    """
    Class to extend a voting rule to ordinal input with a positional scoring rule

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    points : np.array
        the positional scoring rule
    rule : ScoringFunction
        the aggregation rule

    """

    def __init__(self, profile=None,  points=None, rule=None):
        self.profile_ = profile
        if len(points) != self.profile_.m:
            raise ValueError("The positional rule must be of length %i" % (self.profile_.m))
        self.points = points
        if rule != None:
            self.base_rule = rule
        else:
            self.base_rule = None
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = None

    def __call__(self, profile):
        self.profile_ = profile
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = self.base_rule(self.fake_profile_)
        self.delete_cache()
        return self

    def set_rule(self, rule):
        self.base_rule = rule
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = self.base_rule(self.fake_profile_)
        self.delete_cache()
        return self

    def create_fake_profile(self):
        points = np.array(self.points)/np.max(self.points)
        fake_profile = np.zeros((self.profile_.n, self.profile_.m))
        for i in range(self.profile_.n):
            scores_i = self.profile_.scores[i]
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = Profile(self.profile_.m, self.profile_.dim)
        self.profile_.copy(p)
        p.scores = fake_profile
        return p

    def score_(self, cand):
        return self.rule_.scores_[cand]

    def plot_fake_profile(self):
        self.fake_profile_.plot_cands_3D()


class PluralityExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with plurality

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule

    """

    def __init__(self, profile, rule=None):
        m = profile.m
        points = [1] + [0]*(m-1)
        super().__init__(profile, points, rule)


class kApprovalExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with k-approval

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    k : int
        k-Approval parameter
    rule : ScoringFunction
        the aggregation rule

    """
    def __init__(self, profile, k=2, rule=None):
        m = profile.m
        assert(k < m)
        points = [1]*k + [0]*(m-k)
        super().__init__(profile, points, rule)


class VetoExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Veto

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule

    """
    def __init__(self, profile, rule=None):
        m = profile.m
        points = [1]*(m-1) + [0]
        super().__init__(profile, points, rule)

class BordaExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Borda

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule

    """
    def __init__(self, profile, rule=None):
        m = profile.m
        points = [m-i-1 for i in range(m)]
        super().__init__(profile, points, rule)

class InstantRunoffExtension(ScoringFunction):
    """
    Class to extend a voting rule to ordinal input with IRV

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule

    """
    def __init__(self, profile=None,  rule=None):
        self.profile_ = profile
        self.rule_ = rule

    def __call__(self, profile):
        self.profile_ = profile
        self.delete_cache()
        return self

    def set_rule(self, rule):
        self.rule_ = rule
        self.delete_cache()
        return self

    @cached_property
    def ranking_(self):
        m = self.profile_.m
        ranking = np.zeros(m, dtype=int)
        eliminated = []
        for i in range(m):
            fake_profile = self.create_fake_profile(eliminated)
            rule_i = self.rule_(fake_profile)
            loser = rule_i.ranking_[m-1-i]
            ranking[m-i-1] = loser
            eliminated.append(loser)
        return ranking

    def create_fake_profile(self, eliminated):
        fake_profile = np.zeros((self.profile_.n, self.profile_.m))
        points = np.zeros(self.profile_.m)
        points[0] = 1

        for i in range(self.profile_.n):
            scores_i = self.profile_.scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = Profile(self.profile_.m, self.profile_.dim)
        self.profile_.copy(p)
        p.scores = fake_profile
        return p
