# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property


class ScoringRule(DeleteCacheMixin):
    """
    The general class of functions for scoring rules.
    These rules aggregate the scores of every voter to create
    a ranking of the candidates and select a winner.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we run the election.

    Attributes
    ----------
    profile : Profile
        The profile of voters on which we run the election.
    _score_components : int
        The number of components in the aggregated
        score of every candidate. If `> 1`, we
        perform a lexical sort to obtain the ranking.

    """

    def __init__(self, profile=None):
        self.profile_ = profile
        self._score_components = 1

    def __call__(self, profile):
        self.profile_ = profile
        self.delete_cache()
        return self

    def score_(self, candidate):
        """
        Return the aggregated score
        of a given candidate. This should be
        implemented for each scoring rule.

        Parameters
        ----------
        candidate : int
            Index of the candidate for whom we want the score.

        Return
        ------
        float or tuple
            if :attr:`~embedded_voting.ScoringRule._score_components` = 1, return a float,
            otherwise a tuple of length :attr:`~embedded_voting.ScoringRule._score_components`.
        """
        raise NotImplementedError

    @cached_property
    def scores_(self):
        """
        Return the aggregated scores of all candidates.

        Return
        ------
        list
            The scores of all candidates. The score of each
            candidate is a float if :attr:`_score_components` = 1
            and a tuple of length :attr:`_score_components` otherwise.
        """
        return [self.score_(candidate) for candidate in range(self.profile_.n_candidates)]

    @cached_property
    def scores_zip(self):
        """
        Return the scores of all candidates,
        but there is only one component for each candidate.

        When :attr:`_score_components` `> 1`,
        we simply take the last components
        if all other components are maximum,
        and `0` otherwise.

        Return
        ------
        float list
            The scores of every candidates.
        """
        if self._score_components == 1:
            return self.scores_
        else:
            scores = self.scores_
            max_comp = max(scores)
            scores_bis = []
            for s in scores:
                s1 = s[:-1]
                s2 = s[-1]
                if s1 == max_comp[:-1]:
                    scores_bis.append(s2)
                else:
                    scores_bis.append(0)
            return scores_bis

    @cached_property
    def ranking_(self):
        """
        Return the ranking of the candidates
        based on their aggregated scores.

        Return
        ------
        int list
            The ranking of the candidates.
        """

        if self._score_components == 1:
            return list(np.argsort(self.scores_)[::-1])
        else:
            full_scores = []
            for i in range(self._score_components):
                full_scores.append([s[i] for s in self.scores_])
            full_scores = full_scores[::-1]
            return list(np.lexsort(full_scores)[::-1])

    @cached_property
    def winner_(self):
        """
        Return the winner of the election.

        Return
        ------
        int
            The index of the winner of the election.
        """

        return self.ranking_[0]

    @cached_property
    def welfare_(self):
        """
        Return the welfare of all candidates,
        where the welfare is defined as
        `(score - score_min)/(score_max - score_min)`.

        Return
        ------
        float list
            Return the welfare of all candidates.

        """
        scores = self.scores_zip
        max_score = np.max(scores)
        min_score = np.min(scores)
        if max_score == min_score:
            return np.ones(self.profile_.n_candidates)
        return list((scores - min_score) / (max_score - min_score))

    def plot_winner(self, plot_kind="3D", dim=None, fig=None, position=None, show=True):
        """
        Plot the winner of the election.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        fig : matplotlib figure
            The figure on which we add the plot.
        position : list
            The position of the plot on the figure.
            Should be of the form
            ``[n_rows, n_columns, position]``.
        show : bool
            If True, displays the figure
            at the end of the function.

        Return
        ------
        matplotlib ax
            The ax with the plot.

        """
        winner = self.winner_
        ax = self.profile_.plot_candidate(winner, plot_kind=plot_kind, dim=dim, fig=fig, position=position, show=show)
        return ax

    def plot_ranking(self, plot_kind="3D", dim=None, row_size=5, show=True):
        """
        Plot the candidates in the same order than the ranking.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        row_size : int
            Number of subplots by row.
            By default, it is set to 5 by rows.
        show : bool
            If True, displays the figure
            at the end of the function.
        """
        ranking = self.ranking_
        titles = ["#%i. Candidate %i" % (i+1, c) for i, c in enumerate(ranking)]
        self.profile_.plot_candidates(plot_kind=plot_kind, dim=dim, list_candidates=ranking,
                                      list_titles=titles, row_size=row_size, show=show)
