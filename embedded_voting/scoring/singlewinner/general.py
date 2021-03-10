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
    The general class of functions for ScoringRules

    Parameters
    ----------
    profile: Profile
        the profile of voter on which we run the election

    """

    def __init__(self, profile=None):
        self.profile_ = profile
        self.score_components = 1

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
        """
        raise NotImplementedError

    @cached_property
    def scores_(self):
        """
        Return the score of all candidates in the election
        """
        return [self.score_(candidate) for candidate in range(self.profile_.n_candidates)]

    @cached_property
    def ranking_(self):
        """
        Return the ranking of the candidates depending on there scores
        """
        if self.score_components == 1:
            return np.argsort(self.scores_)[::-1]
        else:
            full_scores = []
            for i in range(self.score_components):
                full_scores.append([s[i] for s in self.scores_])
            full_scores = full_scores[::-1]
            return np.lexsort(full_scores)[::-1]

    @cached_property
    def winner_(self):
        """
        Return the winner of the election
        """
        return self.ranking_[0]

    @cached_property
    def welfare_(self):
        """
        Return the welfare of all candidates
        """
        if self.score_components == 1:
            scores = self.scores_
            max_score = np.max(scores)
            min_score = np.min(scores)
            if max_score == min_score:
                return np.ones(self.profile_.n_candidates)
            return (scores - min_score) / (max_score - min_score)
        else:
            scores = self.scores_
            best_score = self.scores_[self.winner_]
            welfare = np.zeros(self.profile_.n_candidates)
            raise ValueError("Not implemented for this rule")

    def plot_winner(self, plot_kind="3D", dim=None, fig=None, position=None, show=True):
        """
        Plot the winner of the election

        Parameters
        __________
        plot_kind : ["3D", "ternary"]
            the kind of plot we want to show.
        dim : array of length 3
            the three dimensions of the embeddings we want to plot.
            default are [0,1,2]
        fig : matplotlib figure or None
            if None, the figure is a default 10x10 matplotlib figure
        position : array of length 3 or None
            the position of the plot on the figure. Default is [1,1,1]
        show : boolean
            if True, execute plt.show() at the end of the function
        """
        winner = self.winner_
        ax = self.profile_.plot_candidate(winner, plot_kind=plot_kind, dim=dim, fig=fig, position=position, show=show)
        return ax

    def plot_ranking(self, plot_kind="3D", dim=None, row_size=5):
        """
        Plot the candidates in the order of the ranking.

        Parameters
        _______
        plot_kind : ["3D", "ternary"]
            the kind of plot we want to show.
        dim : array of length 3
            the three dimensions of the embeddings we want to plot.
            default are [0,1,2]
        row_size : int
            number of figures by row. Default is 5
        """
        ranking = self.ranking_
        titles = ["#%i. Candidate %i" % (i+1, c) for i, c in enumerate(ranking)]
        self.profile_.plot_candidates(plot_kind=plot_kind, dim=dim, list_candidates=ranking,
                                      list_titles=titles, row_size=row_size)
