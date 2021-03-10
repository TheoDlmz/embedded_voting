# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile


class PositionalRuleExtension(ScoringRule):
    """
    Class to extend a voting rule to ordinal input with a positional scoring rule

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    points : np.array
        the vector of the positional scoring rule
    rule : ScoringFunction
        the aggregation rule used
    """

    def __init__(self, profile=None,  points=None, rule=None):
        super().__init__()
        if len(points) != self.profile_.n_candidates:
            raise ValueError("The positional rule must be of length %i" % self.profile_.n_candidates)
        self.points = points
        self.base_rule = rule
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = None
        self.profile_ = profile

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
        fake_profile = np.zeros((self.profile_.n_voters, self.profile_.n_candidates))
        for i in range(self.profile_.n_voters):
            scores_i = self.profile_.scores[i]
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = self.profile_.copy()
        p.scores = fake_profile
        return p

    def score_(self, candidate):
        return self.rule_.scores_[candidate]

    def plot_candidates(self, fake_profile=True,  dim=None, list_candidates=None, list_titles=None, row_size=5):
        if fake_profile:
            self.fake_profile_.plot_candidates(dim=dim, list_candidates=list_candidates,
                                               list_titles=list_titles, row_size=row_size)
        else:
            self.profile_.plot_candidates(dim=dim, list_candidates=list_candidates,
                                          list_titles=list_titles, row_size=row_size)


class PluralityExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Plurality rule

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule used

    """

    def __init__(self, profile, rule=None):
        n_candidates = profile.n_candidates
        points = [1] + [0]*(n_candidates-1)
        super().__init__(profile, points, rule)


class KApprovalExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with k-Approval rule

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    k : int
        k-Approval parameter. Default is 2
    rule : ScoringFunction
        the aggregation rule used

    """
    def __init__(self, profile, k=2, rule=None):
        n_candidates = profile.n_candidates
        assert(k < n_candidates)
        points = [1]*k + [0]*(n_candidates-k)
        super().__init__(profile, points, rule)


class VetoExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Veto rule

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule used

    """
    def __init__(self, profile, rule=None):
        n_candidates = profile.n_candidates
        points = [1]*(n_candidates-1) + [0]
        super().__init__(profile, points, rule)


class BordaExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Borda rule

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule used

    """
    def __init__(self, profile, rule=None):
        n_candidates = profile.n_candidates
        points = [n_candidates-i-1 for i in range(n_candidates)]
        super().__init__(profile, points, rule)


class InstantRunoffExtension(ScoringRule):
    """
    Class to extend a voting rule to ordinal input with Instant Runoff Voting (IRV).
    You cannot get the scores or the welfare of the candidates, because IRV only return a ranking.

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    rule : ScoringFunction
        the aggregation rule

    """

    def __init__(self, profile=None,  rule=None):
        super().__init__(profile)
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

    def score_(self, candidate):
        raise NotImplementedError

    @cached_property
    def ranking_(self):
        n_candidates = self.profile_.n_candidates
        ranking = np.zeros(n_candidates, dtype=int)
        eliminated = []
        for i in range(n_candidates):
            fake_profile = self.create_fake_profile(eliminated)
            rule_i = self.rule_(fake_profile)
            loser = rule_i.ranking_[n_candidates-1-i]
            ranking[n_candidates-i-1] = loser
            eliminated.append(loser)
        return ranking

    def create_fake_profile(self, eliminated):
        fake_profile = np.zeros((self.profile_.n, self.profile_.n_candidates))
        points = np.zeros(self.profile_.n_candidates)
        points[0] = 1

        for i in range(self.profile_.n):
            scores_i = self.profile_.scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = Profile(self.profile_.n_candidates, self.profile_.dim)
        self.profile_.copy(p)
        p.scores = fake_profile
        return p
