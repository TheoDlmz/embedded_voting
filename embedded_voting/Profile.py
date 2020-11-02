# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from embedded_voting.utils import create_3D_plot, normalize


class Profile:
    """
    A class for profiles of embedded voters
    """

    def __init__(self, m, dim):
        self.profile = np.zeros((0, dim))
        self.n = 0
        self.groups = []
        self.n_groups = 0
        self.color_groups = []
        self.scores = np.zeros((0, m))
        self.m = m
        self.dim = dim

    def add_group(self, n_voters, center, scores, dev_center=0.1, dev_scores=0.1, color=""):
        """
        Add a new group of voters
        """
        new_group = np.maximum(0, np.array([center] * n_voters) + np.random.normal(0, 1,
                                                                                   (n_voters, self.dim)) * dev_center)
        new_group = np.array([normalize(x) for x in new_group])

        self.profile = np.concatenate([self.profile, new_group])
        self.groups.extend([self.n_groups] * n_voters)
        self.n_groups += 1
        self.n += n_voters
        new_scores = np.array([scores] * n_voters) + np.random.normal(0, 1, (n_voters, self.m)) * dev_scores
        new_scores = np.minimum(1, new_scores)
        new_scores = np.maximum(0, new_scores)
        self.scores = np.concatenate([self.scores, new_scores])

        if color == "":
            self.color_groups.append([np.random.rand(), np.random.rand(), np.random.rand()])
        else:
            self.color_groups.append(color)

        return self

    def scored_embeddings(self, cand, rc=False):
        embeddings = []
        for i in range(self.n):
            if rc:
                s = np.sqrt(self.scores[i, cand])
            else:
                s = self.scores[i, cand]
            embeddings.append(self.profile[i] * s)
        return np.array(embeddings)

    def plot_profile_3D(self, title="Profile of voters", fig=None, intfig=[1, 1, 1]):
        """
        Plot the profile of voters
        """
        if fig == None:
            fig = plt.figure(figsize=(10, 10))
        ax = create_3D_plot(fig, intfig)
        for i, v in enumerate(self.profile):
            g = self.groups[i]
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=self.color_groups[g], alpha=0.3)
            ax.scatter([v[0]], [v[1]], [v[2]], color='k', s=1)
        ax.set_title(title, fontsize=24)
        plt.show()

    def plot_scores_3D(self, scores, title="", fig=None, intfig=[1, 1, 1], show=True):
        """
        Plot the profile with some scores in 3D space
        """
        if fig == None:
            fig = plt.figure(figsize=(10, 10))
        ax = create_3D_plot(fig, intfig)
        for i, (v, s) in enumerate(zip(self.profile, scores)):
            g = self.groups[i]
            ax.plot([0, s * v[0]], [0, s * v[1]], [0, s * v[2]], color=self.color_groups[g], alpha=0.3)
        ax.set_title(title, fontsize=16)
        if show:
            plt.show()
        return ax

    def plot_cand_3D(self, cand, fig=None, intfig=[1, 1, 1], show=True):
        """
        Plot one candidate of the election in 3D space
        """
        self.plot_scores_3D(self.scores[::, cand], title="Candidate #%i" % (cand + 1), fig=fig, intfig=intfig,
                            show=show)

    def plot_cands_3D(self, list_cand=None, list_titles=None):
        """
        Plot candidates of the elections in a 3D space
        """
        if list_cand == None:
            list_cand = range(self.m)
        if list_titles == None:
            list_titles = ["Candidate %i" % (c + 1) for c in list_cand]
        else:
            list_titles = ["%s (#%i)" % (t, c+1) for (t, c) in zip(list_titles, list_cand)]

        n_cand = len(list_cand)
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(30, n_rows * 5))
        intfig = [n_rows, 6, 1]
        for cand, title in (zip(list_cand, list_titles)):
            self.plot_scores_3D(self.scores[::, cand], title=title, fig=fig, intfig=intfig, show=False)
            intfig[2] += 1

        plt.show()
