# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""

import numpy as np
import matplotlib.pyplot as plt

from embedded_voting.ratings.ratings import Ratings
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.utils.plots import create_3d_plot, create_ternary_plot
from embedded_voting.rules.singlewinner_rules.rule_sum_ratings import RuleSumRatings
from embedded_voting.rules.singlewinner_rules.rule_svd_max import RuleSVDMax
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash
from embedded_voting.embeddings.embeddings import Embeddings


class MovingVoter:
    """
    This subclass of `Embeddings` can be used to see
    what happen to the scores of the different candidates
    when a voter moves from a group to another.

    There is 4 candidates and 3 groups:
    Each group strongly support one of
    the candidate and dislike the other
    candidates, except the last candidate
    which is fine for every group.

    The moving voter is a voter that do not have any
    preference between the candidates
    (she gives a score of `0.8` to every candidate,
    except `0.5` for the last one), but her embeddings
    move from one position to another.

    Parameters
    ----------
    embeddings: Embeddings
        The embeddings of the voters. If none is specified, embeddings are the identity matrix.
    moving_voter: int
        The index of the voter that is moving


    Attributes
    ----------
    rule: Rule
        The rule we are using in the election.
    moving_voter: int
        The index of the voter that is moving.
    ratings_: np.ndarray
        The ratings given by the voters to the candidates


    Examples
    --------
    >>> moving_profile = MovingVoter()
    >>> moving_profile(RuleSumRatings())  # DOCTEST: +ELLIPSIS
    <embedded_voting.experiments.moving_voter.MovingVoter object at ...>
    >>> moving_profile.moving_voter
    0
    >>> moving_profile.embeddings
    Embeddings([[1., 0., 0.],
                [0., 0., 1.],
                [0., 1., 0.],
                [1., 0., 0.]])
    >>> moving_profile.ratings_
    Ratings([[0.8, 0.8, 0.8, 0.5],
             [0.1, 0.1, 1. , 0.5],
             [0.1, 1. , 0.1, 0.5],
             [1. , 0.1, 0.1, 0.5]])
    """
    def __init__(self, embeddings=None, moving_voter=0):
        self.rule = None
        if embeddings is None:
            embeddings = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
        self.embeddings = Embeddings(embeddings, norm=True)
        self.moving_voter = moving_voter
        self.ratings_ = None

    def __call__(self, rule, ratings=None):
        """
        This function is used to update the rule used, and also the ratings

        Parameters
        ----------
        rule: Rule
            The rule we will use
        ratings: np.ndarray or Ratings
            The ratings of the candidates

        Return
        ------
        MovingVoter
            The object
        """
        self.rule = rule
        if ratings is None:
            ratings = np.array([[.8, .8, .8, .5], [.1, .1, 1, .5], [.1, 1, .1, .5], [1, .1, .1, .5]])
        self.ratings_ = Ratings(ratings)
        return self

    def plot_scores_evolution(self, show=True):
        """
        This function plot the evolution
        of the scores of the candidates
        when the moving voters' embeddings
        are changing.

        Parameters
        ----------
        show : bool
            If True, displays the figure
            at the end of the function.

        Examples
        --------
        >>> p = MovingVoter()(RuleSVDNash())
        >>> p.plot_scores_evolution(show=False)
        """

        tab_x = np.linspace(0, 1, 50)
        tab_y = []
        for x in tab_x:
            self.embeddings[self.moving_voter] = normalize([1-x, x, 0])
            tab_y.append(self.rule(self.ratings_, self.embeddings).scores_focus_on_last_)

        tab_y = np.array(tab_y).T
        name = ["Start", "End", "Orthogonal", "Consensus"]
        for i in range(self.ratings_.n_candidates):
            plt.plot(tab_x, tab_y[i], label=name[i])

        plt.title("Evolution of the score")
        plt.xlabel("X coordinate of moving voter")
        plt.ylabel("Score")
        plt.xlim(0, 1)
        plt.legend()
        if show:
            plt.show()  # pragma: no cover

    def plot_features_evolution(self, show=True):
        """
        This function plot the evolution
        of the features of the candidates
        when the moving voters' embeddings
        are changing. Only works for :class:`RuleSVDMax` and :class:`RuleFeatures`.

        Parameters
        ----------
        show : bool
            If True, displays the figure
            at the end of the function.

        Examples
        --------
        >>> p = MovingVoter()(RuleSVDMax())
        >>> p.plot_features_evolution(show=False)
        """
        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.embeddings[self.moving_voter] = normalize([1-x, x, 0])
            tab_y.append(self.rule(self.ratings_, self.embeddings).features_)
        tab_y = np.array(tab_y)

        fig = plt.figure(figsize=(10, 5))

        ax = create_3d_plot(fig, position=[1, 2, 1])
        name = ["Start", "End", "Orthogonal", "Consensus"]
        for i in range(self.ratings_.n_candidates):
            vec_init = normalize(tab_y[0, i])**2
            ax.plot(tab_y[::, i, 0], tab_y[::, i, 1], tab_y[::, i, 2],
                    color=(vec_init[0] * 0.8, vec_init[1] * 0.8, vec_init[2] * 0.8),
                    alpha=0.5, label=name[i])
            for j, v in enumerate(tab_y[::, i]):
                vec_normalized = normalize(v)
                ax.plot([0, v[0]], [0, v[1]], [0, v[2]],
                        color=(vec_normalized[0] * 0.8, vec_normalized[1] * 0.8, vec_normalized[2] * 0.8),
                        alpha=j / 60)
        ax.set_title("Evolution of the features")
        plt.legend()

        tax = create_ternary_plot(fig, position=[1, 2, 2])
        for i in range(self.ratings_.n_candidates):
            points = [normalize(x[[0, 2, 1]])**2 for x in tab_y[::, i]]
            vec_init = normalize(tab_y[0, i, [0, 2, 1]])**2
            tax.plot(points, color=(vec_init[0] * 0.8, vec_init[2] * 0.8, vec_init[1] * 0.8), alpha=0.8)
            tax.scatter([vec_init],
                        color=(vec_init[0] * 0.8, vec_init[2] * 0.8, vec_init[1] * 0.8),
                        alpha=0.7, s=50)

        if show:
            plt.show()  # pragma: no cover
