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


class MovingVoter(DeleteCacheMixin):
    """
    A class to see what happen to the scores when a voter moves from a group to another

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
            self.profile_.embs[3] = normalize([1-x, x, 0])
            tab_y.append(self.rule_(self.profile_).scores_)

        tab_y = np.array(tab_y).T
        name = ["Start", "End", "Orth", "Consensus"]
        for i in range(4):
            plt.plot(tab_x, tab_y[i], label=name[i])

        plt.title("Evolution of the scoring")
        plt.xlabel("x coordinate of moving voter")
        plt.ylabel("Score")
        plt.xlim(0, 1)
        plt.legend()
        plt.show()


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
