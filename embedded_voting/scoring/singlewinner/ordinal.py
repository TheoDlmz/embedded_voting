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
from embedded_voting.scoring.singlewinner.svd import *


class PositionalRuleExtension(ScoringRule):
    """
    Class to extend a voting rule to ordinal input with a positional scoring rule.

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we run the election
    points : list
        The vector of the positional scoring rule.
    rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.

    Attributes
    ----------
    profile_ : Profile
        The profile of voter.
    fake_profile_ : Profile
        The modified profile of voter (with ordinal scores) on which we run the election.
    points : np.ndarray
        The vector of the positional scoring rule.
    base_rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.
    _rule : ScoringRule
        The aggregation rule with the fake profile.
    _score_components :
        The number of components in the score of every candidate. If > 1, we do a lexical sort.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDNash(my_profile)
    >>> election.ranking_
    [3, 0, 1, 2]
    >>> election_bis = PositionalRuleExtension(my_profile, [2, 1, 1, 0])
    >>> election_bis.fake_profile_.scores
    array([[0. , 0.5, 0.5, 1. ],
           [0.5, 1. , 0.5, 0. ],
           [1. , 0.5, 0. , 0.5]])
    >>> election_bis.set_rule(SVDNash()).ranking_
    [1, 3, 0, 2]
    """

    def __init__(self, profile,  points, rule=None):
        super().__init__(profile)
        if len(points) != self.profile_.n_candidates:
            raise ValueError("The positional rule must be of length %i" % self.profile_.n_candidates)
        self.points = points
        self.base_rule = rule
        if rule is not None:
            self._score_components = rule.score_components
        self.fake_profile_ = self._create_fake_profile()
        self._rule = None

    def __call__(self, profile):
        self.profile_ = profile
        self.fake_profile_ = self._create_fake_profile()
        if self.base_rule is not None:
            self._rule = self.base_rule(self.fake_profile_)
        self.delete_cache()
        return self

    def set_rule(self, rule):
        """
        This function updates the aggregation rule used for the election.

        Parameters
        ----------
        rule : ScoringRule
            The new rule to use.

        Return
        ------
        PositionalRuleExtension
            The object itself
        """
        self.base_rule = rule
        self.fake_profile_ = self._create_fake_profile()
        self._rule = self.base_rule(self.fake_profile_)
        self._score_components = rule._score_components
        self.delete_cache()
        return self

    def _create_fake_profile(self):
        """
        This function creates the fake profile for the election (using the points vector).

        Return
        ------
        Profile
            The fake profile
        """
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
        return self._rule.scores_[candidate]

    def plot_fake_profile(self, plot_kind="3D", dim=None, list_candidates=None, list_titles=None, row_size=5, show=True):
        """
        This function plot the candidate in the fake profile, using the scoring vector :attr:`points`.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        dim : list
            The 3 dimensions we are using for our plot.
        list_candidates : int list
            The list of candidates we want to plot. Should contains integer lower than
            :attr:`n_candidates`. Default is range(:attr:`n_candidates`).
        list_titles : str list
            Contains the title of the plots.Should be the same length than list_candidates.
        row_size : int
            Number of subplots by row. Default is set to 5.
        show : bool
            If True, plot the figure at the end of the function.

        Return
        ------
        Profile
            The fake profile
        """
        self.fake_profile_.plot_candidates(plot_kind=plot_kind, dim=dim, list_candidates=list_candidates,
                                           list_titles=list_titles, row_size=row_size, show=show)


class PluralityExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Plurality (vector `[1, 0, ..., 0]`)

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we run the election
    rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = PluralityExtension(my_profile)
    >>> election.fake_profile_.scores
    array([[0., 0., 0., 1.],
           [0., 1., 0., 0.],
           [1., 0., 0., 0.]])
    >>> election.set_rule(SVDNash(use_rank=True)).ranking_
    [3, 1, 0, 2]
    """

    def __init__(self, profile, rule=None):
        n_candidates = profile.n_candidates
        points = [1] + [0]*(n_candidates-1)
        super().__init__(profile, points, rule)


class KApprovalExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with k-Approval (vector `[1, 1, 0,..., 0]`) with `k`
    1 at the beginning of the vector.

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we run the election
    k : int
        The k parameter of the k-approval. By default, it is set to 2.
    rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = KApprovalExtension(my_profile)
    >>> election.fake_profile_.scores
    array([[0., 0., 1., 1.],
           [0., 1., 1., 0.],
           [1., 1., 0., 0.]])
    >>> election.set_rule(SVDNash(use_rank=True)).ranking_
    [1, 2, 3, 0]
    """
    def __init__(self, profile, k=2, rule=None):
        n_candidates = profile.n_candidates
        assert(k < n_candidates)
        points = [1]*k + [0]*(n_candidates-k)
        super().__init__(profile, points, rule)


class VetoExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Veto (vector `[1, 1,..., 1, 0]`)

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we run the election
    rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = VetoExtension(my_profile)
    >>> election.fake_profile_.scores
    array([[0., 1., 1., 1.],
           [1., 1., 1., 0.],
           [1., 1., 0., 1.]])
    >>> election.set_rule(SVDNash()).ranking_
    [1, 3, 0, 2]
    """
    def __init__(self, profile, rule=None):
        n_candidates = profile.n_candidates
        points = [1]*(n_candidates-1) + [0]
        super().__init__(profile, points, rule)


class BordaExtension(PositionalRuleExtension):
    """
    Class to extend a voting rule to ordinal input with Borda (vector `[m-1, m-2..., 1, 0]`) where `m`
    is defined as :attr:`n_candidates`.

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we run the election
    rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = BordaExtension(my_profile)
    >>> election.fake_profile_.scores
    array([[0.        , 0.33333333, 0.66666667, 1.        ],
           [0.33333333, 1.        , 0.66666667, 0.        ],
           [1.        , 0.66666667, 0.        , 0.33333333]])
    >>> election.set_rule(SVDNash()).ranking_
    [1, 3, 2, 0]
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
    ----------
    profile : Profile
        The profile of voter on which we run the election
    rule : ScoringFunction
        The aggregation rule used to determine the scores of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = InstantRunoffExtension(my_profile)
    >>> election.set_rule(SVDNash()).ranking_
    [1, 3, 2, 0]
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
            fake_profile = self._create_fake_profile(eliminated)
            rule_i = self.rule_(fake_profile)
            loser = rule_i.ranking_[n_candidates-1-i]
            ranking[n_candidates-i-1] = loser
            eliminated.append(loser)
        return list(ranking)

    def _create_fake_profile(self, eliminated):
        """
        This function creates the fake profile for the election (using the points vector).

        Return
        ------
        Profile
            The fake profile
        """
        fake_profile = np.zeros((self.profile_.n_voters, self.profile_.n_candidates))
        points = np.zeros(self.profile_.n_candidates)
        points[0] = 1

        for i in range(self.profile_.n_voters):
            scores_i = self.profile_.scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = self.profile_.copy()
        p.scores = fake_profile
        return p
