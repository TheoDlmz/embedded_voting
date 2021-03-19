# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.profile.Profile import Profile
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.utils.plots import create_3D_plot, create_ternary_plot
from embedded_voting.scoring.singlewinner.trivialRules import SumScores
from embedded_voting.scoring.singlewinner.svd import SVDMax, SVDNash


class MovingVoterProfile(Profile):
    """
    This subclass of `Profile` can be used to see
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
    rule : ScoringFunction
        The rule we are using in the election.

    Attributes
    ----------
    rule_ : ScoringFunction
        The rule we are using in the election.
    moving_voter : int
        The index of the voter that is moving.

    Examples
    --------
    >>> moving_profile = MovingVoterProfile()
    >>> moving_profile(SumScores())
    <embedded_voting.profile.MovingVoter.MovingVoterProfile object at ...>
    >>> moving_profile.moving_voter
    0
    >>> moving_profile.embeddings
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    >>> moving_profile.scores
    array([[0.8, 0.8, 0.8, 0.5],
           [0.1, 0.1, 1. , 0.5],
           [0.1, 1. , 0.1, 0.5],
           [1. , 0.1, 0.1, 0.5]])
    """
    def __init__(self, rule=None):
        super().__init__(4, 3)
        self.rule_ = rule
        self.add_voter([1, 0, 0], [.8, .8, .8, .5])
        self.add_voter([0, 0, 1], [.1, .1, 1, .5])
        self.add_voter([0, 1, 0], [.1, 1, .1, .5])
        self.add_voter([1, 0, 0], [1, .1, .1, .5])
        self.moving_voter = 0

    def __call__(self, r):
        self.rule_ = r
        return self

    def set_moving_voter(self, voter):
        """
        This function update the
        index of the moving voter :attr:`moving_voter`.

        Parameters
        ----------
        voter : int
            The index of the new moving voter.

        Return
        ------
        MovingVoterProfile
            The object itself.

        Examples
        --------
        >>> moving_profile = MovingVoterProfile()
        >>> moving_profile.moving_voter
        0
        >>> moving_profile.set_moving_voter(3)
        <embedded_voting.profile.MovingVoter.MovingVoterProfile object at ...>
        >>> moving_profile.moving_voter
        3
        """
        self.moving_voter = voter
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
        >>> p = MovingVoterProfile(SVDNash())
        >>> p.plot_scores_evolution(show=False)
        """

        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.embeddings[self.moving_voter] = normalize([1-x, x, 0])
            tab_y.append(self.rule_(self).scores_zip)

        tab_y = np.array(tab_y).T
        name = ["Start", "End", "Orthogonal", "Consensus"]
        for i in range(self.n_candidates):
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
        are changing. Only works for :class:`SVDMax` and :class:`FeaturesRule`.

        Parameters
        ----------
        show : bool
            If True, displays the figure
            at the end of the function.

        Examples
        --------
        >>> p = MovingVoterProfile(SVDMax())
        >>> p.plot_features_evolution(show=False)
        """
        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.embeddings[self.moving_voter] = normalize([1-x, x, 0])
            tab_y.append(self.rule_(self).features_)
        tab_y = np.array(tab_y)

        fig = plt.figure(figsize=(10, 5))

        ax = create_3D_plot(fig, position=[1, 2, 1])
        name = ["Start", "End", "Orthogonal", "Consensus"]
        for i in range(self.n_candidates):
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
        for i in range(self.n_candidates):
            points = [normalize(x[[0, 2, 1]])**2 for x in tab_y[::, i]]
            vec_init = normalize(tab_y[0, i, [0, 2, 1]])**2
            tax.plot(points, color=(vec_init[0] * 0.8, vec_init[2] * 0.8, vec_init[1] * 0.8), alpha=0.8)
            tax.scatter([vec_init],
                        color=(vec_init[0] * 0.8, vec_init[2] * 0.8, vec_init[1] * 0.8),
                        alpha=0.7, s=50)

        if show:
            plt.show()  # pragma: no cover
