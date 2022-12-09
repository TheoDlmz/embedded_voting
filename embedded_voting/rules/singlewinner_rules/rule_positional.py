# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_identity import EmbeddingsFromRatingsIdentity


class RulePositional(Rule):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with a positional scoring rule.

    Parameters
    ----------
    points : list
        The vector of the positional scoring rule.
        Should be of the same length than the number
        of candidates. In each ranking, candidate ranked
        at position `i` get `points[i]` points.
    rule : Rule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

    Attributes
    ----------
    fake_ratings_ : ratings
        The modified ratings of voters (with ordinal scores)
        on which we run the election.
    points : np.ndarray
        The vector of the positional scoring rule.
        Should be of the same length than the number
        of candidates. In each ranking, candidate ranked
        at position `i` get `points[i]` points.
    base_rule : Rule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.
    _rule : Rule
        The aggregation rule instantiated
        with the :attr:`fake_ratings`.
    _score_components :
        The number of components in the score
        of every candidate. If `> 1`,
        we perform a lexical sort to obtain
        the ranking.

    Examples
    --------
    >>> ratings = np.array([[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]])
    >>> embeddings = Embeddings([[1, 0], [1, 1], [0, 1]], norm=True)
    >>> election = RuleSVDNash()(ratings, embeddings)
    >>> election.ranking_
    [3, 0, 1, 2]
    >>> election_bis = RulePositional([2, 1, 1, 0])(ratings, embeddings)
    >>> election_bis.fake_ratings_
    Ratings([[0. , 0.5, 0.5, 1. ],
             [0.5, 1. , 0.5, 0. ],
             [1. , 0.5, 0. , 0.5]])
    >>> election_bis.set_rule(RuleSVDNash())(ratings, embeddings).ranking_
    [1, 3, 0, 2]
    """

    def __init__(self, points, rule=None):
        super().__init__()
        self.points = points
        self.base_rule = rule
        if rule is not None:
            self.score_components = rule.score_components
        self._rule = None

    def __call__(self, ratings, embeddings=None):
        self.ratings_ = Ratings(ratings)
        if embeddings is None:
            embeddings = EmbeddingsFromRatingsIdentity()(self.ratings_)
        self.embeddings_ = Embeddings(embeddings, norm=True)
        self.fake_ratings_ = Ratings(self._create_fake_ratings())
        if self.base_rule is not None:
            self._rule = self.base_rule(self.fake_ratings_, self.embeddings_)
        self.delete_cache()
        return self

    def set_rule(self, rule):
        """
        This function updates the :attr:`base_rule` used for the election.

        Parameters
        ----------
        rule : Rule
            The new rule to use.

        Return
        ------
        RulePositional
            The object itself.
        """
        self.base_rule = rule
        self.score_components = rule.score_components
        self.delete_cache()
        return self

    def _create_fake_ratings(self):
        """
        This function creates the
        fake ratings for the election
        (using the :attr:`points` vector).

        Return
        ------
        np.ndarray
            The fake ratings.
        """
        points = np.array(self.points)/np.max(self.points)
        fake_ratings = np.zeros(self.ratings_.shape)
        for i in range(self.ratings_.n_voters):
            scores_i = self.ratings_.voter_ratings(i)
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_ratings[i] = points[ord_i]

        return fake_ratings

    def _score_(self, candidate):
        return self._rule.scores_[candidate]

    def plot_fake_ratings(self, plot_kind="3D", dim=None, list_candidates=None,
                          list_titles=None, row_size=5, show=True):
        """
        This function plot the candidates
        in the fake ratings, obtained using the
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
            ``n_candidates``. By default, we
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

        """
        self.embeddings_.plot_candidates(self.fake_ratings_,
                                         plot_kind=plot_kind,
                                         dim=dim,
                                         list_candidates=list_candidates,
                                         list_titles=list_titles,
                                         row_size=row_size,
                                         show=show)
