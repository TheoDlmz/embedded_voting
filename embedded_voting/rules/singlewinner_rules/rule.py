# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.utils.miscellaneous import ranking_from_scores, winner_from_scores


class Rule(DeleteCacheMixin):
    """
    The general class of functions for scoring rules.
    These rules aggregate the scores of every voter to create
    a ranking of the candidates and select a winner.

    Parameters
    ----------
    score_components : int
        The number of components in the aggregated
        score of every candidate. If `> 1`, we
        perform a lexical sort to obtain the ranking.
    embeddings_from_ratings: EmbeddingsFromRatings
        If no embeddings are specified in the call, this `EmbeddingsFromRatings` object is use to generate
        the embeddings from the ratings. Default: `EmbeddingsFromRatingsIdentity()`.

    Attributes
    ----------
    ratings_ : Ratings
        The ratings of voters on which we run the election.
    embeddings_ : Embeddings
        The embeddings of the voters on which we run the election.
    """

    def __init__(self, score_components=1, embeddings_from_ratings=None):
        self.score_components = score_components
        self.embeddings_from_ratings = embeddings_from_ratings
        self.ratings_ = None
        self.embeddings_ = None

    def __call__(self, ratings, embeddings=None):
        """
        Parameters
        ----------
        ratings : Ratings or list or np.ndarray
            The ratings of voters on which we run the election.
        embeddings : Embeddings or list or np.ndarray
            The embeddings of the voters on which we run the election.

        Return
        ------
        Rule
            The object itself.
        """
        self.delete_cache()
        self.ratings_ = Ratings(ratings)
        if embeddings is not None:
            self.embeddings_ = Embeddings(embeddings, norm=False)
        elif self.embeddings_from_ratings is not None:
            self.embeddings_ = self.embeddings_from_ratings(self.ratings_)
        else:
            # Useful for rules that do not rely on the embeddings, such as RuleSumRatings.
            self.embeddings_ = None
        return self

    def _score_(self, candidate):
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
            if :attr:`score_components` = 1, return a float,
            otherwise a tuple of length :attr:`score_components`.
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
            candidate is a float if :attr:`score_components` = 1
            and a tuple of length :attr:`score_components` otherwise.
        """
        return [self._score_(candidate) for candidate in range(self.ratings_.n_candidates)]

    def score_(self, candidate):
        """
        Return the aggregated score of a given candidate.

        Parameters
        ----------
        candidate : int
            Index of the candidate for whom we want the score.

        Return
        ------
        float or tuple
            if :attr:`score_components` = 1, return a float,
            otherwise a tuple of length :attr:`score_components`.
        """
        # Note for developers: this method is called by the user to prevent from calling _score_ every time.
        return self.scores_[candidate]

    @cached_property
    def scores_focus_on_last_(self):
        """
        Return the last score component of each candidate, but only if the other score components are maximal.

        If :attr:`score_components` is 1, return :attr:`scores_`. Otherwise, for each candidate:

        * Return the last score component if all other components are maximal.
        * Return 0 otherwise.

        Note that if the last score component is defined as non-negative, and if it is always positive for the winner,
        then :attr:`scores_focus_on_last_` is enough to determine which candidate has the best score by lexicographical
        order.

        Return
        ------
        float list
            The scores of every candidates.

        Examples
        --------
        Cf. :class:`RuleMaxParallelepiped`.
        """
        if self.score_components == 1:
            return self.scores_
        else:
            max_comp = max(self.scores_)
            return [s[-1] if s[:-1] == max_comp[:-1] else 0 for s in self.scores_]

    @cached_property
    def ranking_(self):
        """
        Return the ranking of the candidates based on their aggregated scores.

        Return
        ------
        list of int
            The ranking of the candidates. In case of tie, candidates with lower indices are favored.
        """
        return ranking_from_scores(self.scores_)

    @cached_property
    def winner_(self):
        """
        Return the winner of the election.

        Return
        ------
        int
            The index of the winner of the election. In case of tie, candidates with lower indices are favored.
        """
        return winner_from_scores(self.scores_)

    @cached_property
    def welfare_(self):
        """
        Return the welfare of all candidates, where the welfare is defined as
        `(score - score_min)/(score_max - score_min)`.

        If scores are tuple, then `scores_focus_on_last_` is used.

        If `score_max = score_min`, then by convention, all candidates have a welfare of 1.

        Return
        ------
        list of float
            Welfare of all candidates.
        """
        max_score = np.max(self.scores_focus_on_last_)
        min_score = np.min(self.scores_focus_on_last_)
        if max_score == min_score:
            return [1.] * self.ratings_.n_candidates
        return list((self.scores_focus_on_last_ - min_score) / (max_score - min_score))

    def plot_winner(self, plot_kind="3D", dim=None, fig=None, plot_position=None, show=True):
        """
        Plot the matrix associated to the winner of the election.

        Cf. :meth:`Embeddings.plot_candidate`.

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
        plot_position : list
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
        ax = self.embeddings_.plot_candidate(ratings=self.ratings_,
                                             candidate=self.winner_,
                                             plot_kind=plot_kind,
                                             dim=dim,
                                             fig=fig,
                                             plot_position=plot_position,
                                             show=show)
        return ax

    def plot_ranking(self, plot_kind="3D", dim=None, row_size=5, show=True):
        """
        Plot the matrix associated to each candidate, in the same order than the ranking of the election.

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
        self.embeddings_.plot_candidates(ratings=self.ratings_,
                                         plot_kind=plot_kind,
                                         dim=dim,
                                         list_candidates=ranking,
                                         list_titles=titles,
                                         row_size=row_size,
                                         show=show)
