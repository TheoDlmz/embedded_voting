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
    The general class of functions for scoring rules. These rules aggregate the scores of every voter to create
    a ranking of the candidate and select a winner.

    Parameters
    ----------
    profile : Profile
        the profile of voter on which we run the election

    Attributes
    ----------
    profile : Profile
        The profile of voter on which we run the election
    _score_components : int
        The number of components in the score of every candidate. If > 1, we do a lexical sort.

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
        Return the score of a given candidate. Need to be implemented for each scoring rule.

        Parameters
        ----------
        candidate : int
            Index of the candidate for which we want the score

        Return
        ------
        float or tuple
            if :attr:`_score_component` = 1, return a float, otherwise a tuple of length :attr:`_score_component`
        """
        raise NotImplementedError

    @cached_property
    def scores_(self):
        """
        Return the scores of all candidates in the election.

        Return
        ------
        list
            The score of every candidates. The score of each candidate is a float if :attr:`_score_component` = 1
            and a tuple of length :attr:`_score_component` otherwise
        """
        return [self.score_(candidate) for candidate in range(self.profile_.n_candidates)]

    @cached_property
    def ranking_(self):
        """
        Return the ranking of the candidates depending on there scores.

        Return
        ------
        int list
            The ranking of the candidates
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
            The index of the winner of the election
        """
        return self.ranking_[0]

    @cached_property
    def welfare_(self):
        """
        Return the welfare of all candidates.

        Return
        ------
        float list
            Return the welfare of every candidate, where the welfare is defined as
            `(score - score_min)/(score_max - score_min)`. Not defined if :attr:`_score_components` > 1.

        """
        if self._score_components == 1:
            scores = self.scores_
            max_score = np.max(scores)
            min_score = np.min(scores)
            if max_score == min_score:
                return np.ones(self.profile_.n_candidates)
            return (scores - min_score) / (max_score - min_score)
        else:
            raise NotImplementedError

    def plot_winner(self, plot_kind="3D", dim=None, fig=None, position=None, show=True):
        """
        Plot the winner of the election.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        dim : list
            The 3 dimensions we are using for our plot.
        fig : matplotlib figure
            The figure on which we add the plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.
        show : bool
            If True, show the figure at the end of the function

        Return
        ------
        matplotlib ax
            The ax with the plot

        """
        winner = self.winner_
        ax = self.profile_.plot_candidate(winner, plot_kind=plot_kind, dim=dim, fig=fig, position=position, show=show)
        return ax

    def plot_ranking(self, plot_kind="3D", dim=None, row_size=5, show=True):
        """
        Plot the candidates in the order of the ranking.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        dim : list
            The 3 dimensions we are using for our plot.
        row_size : int
            Number of subplots by row. Default is set to 5.
        show : bool
            If True, plot the figure at the end of the function.
        """
        ranking = self.ranking_
        titles = ["#%i. Candidate %i" % (i+1, c) for i, c in enumerate(ranking)]
        self.profile_.plot_candidates(plot_kind=plot_kind, dim=dim, list_candidates=ranking,
                                      list_titles=titles, row_size=row_size, show=show)
