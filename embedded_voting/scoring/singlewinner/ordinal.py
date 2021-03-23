# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile
from embedded_voting.scoring.singlewinner.svd import *


class PositionalRuleExtension(ScoringRule):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with a positional scoring rule.

    Parameters
    ----------
    profile : Profile
        The profile of voters on
        which we run the election.
    points : list
        The vector of the positional scoring rule.
        Should be of the same length than the number
        of candidates. In each ranking, candidate ranked
        at position `i` get `points[i]` points.
    rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

    Attributes
    ----------
    profile_ : Profile
        The profile of voters.
    fake_profile : Profile
        The modified profile of voters (with ordinal scores)
        on which we run the election.
    points : np.ndarray
        The vector of the positional scoring rule.
        Should be of the same length than the number
        of candidates. In each ranking, candidate ranked
        at position `i` get `points[i]` points.
    base_rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.
    _rule : ScoringRule
        The aggregation rule instantiated
        with the :attr:`fake_profile`.
    _score_components :
        The number of components in the score
        of every candidate. If `> 1`,
        we perform a lexical sort to obtain
        the ranking.

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
    >>> election_bis.fake_profile.scores
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
            self._score_components = rule._score_components
        self.fake_profile = self._create_fake_profile()
        self._rule = None
        self(profile)

    def __call__(self, profile):
        self.profile_ = profile
        self.fake_profile = self._create_fake_profile()
        if self.base_rule is not None:
            self._rule = self.base_rule(self.fake_profile)
        self.delete_cache()
        return self

    def set_rule(self, rule):
        """
        This function updates the :attr:`base_rule` used for the election.

        Parameters
        ----------
        rule : ScoringRule
            The new rule to use.

        Return
        ------
        PositionalRuleExtension
            The object itself.
        """
        self.base_rule = rule
        self.fake_profile = self._create_fake_profile()
        self._rule = self.base_rule(self.fake_profile)
        self._score_components = rule._score_components
        self.delete_cache()
        return self

    def _create_fake_profile(self):
        """
        This function creates the
        fake profile for the election
        (using the :attr:`points` vector).

        Return
        ------
        Profile
            The fake profile.
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

    def plot_fake_profile(self, plot_kind="3D", dim=None, list_candidates=None,
                          list_titles=None, row_size=5, show=True):
        """
        This function plot the candidates
        in the fake profile, obtained using the
        scoring vector :attr:`points`.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        list_candidates : int list
            The list of candidates we want to plot.
            Should contains integers lower than
            :attr:`~embedded_voting.Profile.n_candidates`. By default, we
            plot all candidates.
        list_titles : str list
            Contains the title of the plots.
            Should be the same length than `list_candidates`.
        row_size : int
            Number of subplots by row.
            By default, it is set to 5 plots by rows.
        show : bool
            If True, displays the figure
            at the end of the function.

        Return
        ------
        Profile
            The fake profile.
        """
        self.fake_profile.plot_candidates(plot_kind=plot_kind, dim=dim, list_candidates=list_candidates,
                                          list_titles=list_titles, row_size=row_size, show=show)


class PluralityExtension(PositionalRuleExtension):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with Plurality rule (vector ``[1, 0, ..., 0]``).

    Parameters
    ----------
    profile : Profile
        The profile of voters on
        which we run the election.
    rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = PluralityExtension(my_profile)
    >>> election.fake_profile.scores
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
    This class enables to extend a
    voting rule to an ordinal input
    with k-Approval rule (vector ``[1, 1, ..., 0]``
    with `k` ones).

    Parameters
    ----------
    profile : Profile
        The profile of voters on
        which we run the election.
    k : int
        The k parameter of the k-approval.
        By default, it is set to 2.
    rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.


    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = KApprovalExtension(my_profile)
    >>> election.fake_profile.scores
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
    This class enables to extend a
    voting rule to an ordinal input
    with Veto rule (vector ``[1, ..., 1, 0]``).

    Parameters
    ----------
    profile : Profile
        The profile of voters on
        which we run the election.
    rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = VetoExtension(my_profile)
    >>> election.fake_profile.scores
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
    This class enables to extend a
    voting rule to an ordinal input
    with Borda rule (vector ``[m-1, m-2, ..., 1, 0]``).

    Parameters
    ----------
    profile : Profile
        The profile of voters on
        which we run the election.
    rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = BordaExtension(my_profile)
    >>> election.fake_profile.scores
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
    This class enables to extend a
    voting rule to an ordinal input
    with Instant Runoff ranking. You cannot access
    to the :attr:`~embedded_voting.ScoringRule.scores_` because IRV only
    compute the ranking of the candidates.

    Parameters
    ----------
    profile : Profile
        The profile of voters on
        which we run the election.
    rule : ScoringRule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

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
        This function creates a fake profile for the election, based
        on the candidates already eliminated during the previous
        steps.

        Return
        ------
        Profile
            The fake profile.
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
