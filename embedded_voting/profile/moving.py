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
from embedded_voting.utils.plots import create_3D_plot
from embedded_voting.utils.cached import DeleteCacheMixin
from embedded_voting.scoring.singlewinner.trivialRules import SumScores


class MovingVoterProfile(Profile):
    """
    A class to see what happen to the scores of the different candidates when a voter moves from a group to another.
    There is 4 candidates and 3 groups: Each group strongly support one of the candidate and dislike the other
    candidates, except the last candidate which is okay for everyone. The moving voter is a voter that do not have any
    preference between the candidates (she gives a score to 0.75 to every candidate) but her embeddings move from
    one position to another

    Parameters
    ----------
    rule : ScoringFunction
        The rule we are using in this election

    Attributes
    ----------
    rule_ : ScoringFunction
        The rule we are using in this election
    moving_voter : int
        The index of the voter which is moving

    Examples
    --------
    >>> moving_profile = MovingVoterProfile()
    >>> moving_profile(SumScores())
    <embedded_voting.profile.moving.MovingVoterProfile object at ...>
    >>> moving_profile.moving_voter
    0
    >>> moving_profile.embeddings
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    >>> moving_profile.scores
    array([[0.5, 0.5, 0.5, 0.5],
           [0.1, 0.1, 0.9, 0.5],
           [0.1, 0.9, 0.1, 0.5],
           [0.9, 0.1, 0.1, 0.5]])
    """
    def __init__(self, rule=None):
        super().__init__(4, 3)
        self.rule_ = rule
        self.add_voter([1, 0, 0], [.5, .5, .5, .5])
        self.add_voter([0, 0, 1], [.1, .1, .9, .5])
        self.add_voter([0, 1, 0], [.1, .9, .1, .5])
        self.add_voter([1, 0, 0], [.9, .1, .1, .5])
        self.moving_voter = 0

    def __call__(self, r):
        self.rule_ = r
        return self

    def set_moving_voter(self, voter):
        """
        This function change the attribute :attr:`moving_voter`

        Parameters
        ----------
        voter : int
            The index of the new moving voter

        Return
        ------
        MovingVoterProfile
            The object itself

        Examples
        --------
        >>> moving_profile = MovingVoterProfile()
        >>> moving_profile.moving_voter
        0
        >>> moving_profile.set_moving_voter(3)
        <embedded_voting.profile.moving.MovingVoterProfile object at ...>
        >>> moving_profile.moving_voter
        3
        """
        self.moving_voter = voter
        return self

    def plot_evolution(self, show=True):
        """
        This function plot the evolution of the scores of the candidates when the moving voters' embeddings
        are changing.

        Parameters
        ----------
        show : bool
            If True, the figure is shown at the end of the function

        """
        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.embeddings[self.moving_voter] = normalize([1-x, x, 0])
            tab_y.append(self.rule_(self).scores_)

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
            plt.show()  # pragma : no cover


class MovingVoterFeatures(DeleteCacheMixin):
    """
    A class to see what happen to the featyres when a voter moves from a group to another

    Parameters
    ________
    r : ScoringFunction
        the rule we are using
    """
    def __init__(self, r=None):
        self.rule_ = None
        self.profile_ = Profile(4, 3)
        self.profile_.add_group(1, [0, 0, 1], [0, 0, 1, 0.5], 0, 0)
        self.profile_.add_group(1, [0, 1, 0], [0, 1, 0, 0.5], 0, 0)
        self.profile_.add_group(1, [1, 0, 0], [1, 0, 0, 0.5], 0, 0)
        self.profile_.add_group(1, [1, 0, 0], [0.75, 0.75, 0.75, 0.75], 0, 0)
        if r is not None:
            self(r)

    def __call__(self, r):
        self.rule_ = r
        return self

    def plot_evol(self):
        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.profile_.embs[3] = normalize([1 - x, x, 0])
            _, vectors = self.rule_(self.profile_).winner_k()
            tab_y.append(vectors)
        tab_y = np.array(tab_y)

        fig = plt.figure(figsize=(10, 10))
        ax = create_3D_plot(fig)
        colors = ["blue", "orange", "green", "red"]
        name = ["Start", "End", "Orth", "Consensus"]
        for i in range(4):
            ax.plot(tab_y[::, i, 0], tab_y[::, i, 1], tab_y[::, i, 2], color=colors[i], alpha=0.5, label=name[i])
            for j, v in enumerate(tab_y[::, i]):
                ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=colors[i], alpha=j/60)
        plt.legend()
        plt.show()
