# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.utils.plots import create_2D_plot, create_3D_plot


class Profile(DeleteCacheMixin):
    """
    A profile of voter

    Parameters
    _________
    m : int
        The number of candidate for this profile
    dim : int
        The number of dimensions for the voters' embeddings

    Attributes
    ___________
    n : int
        The number of voter of the profile
    embs : n x dim numpy array
        The embeddings of the voters
    scoring : n x m numpy array
        The scoring of the candidates
    """

    def __init__(self, m, dim):
        self.embs = np.zeros((0, dim))
        self.scores = np.zeros((0, m))
        self.m = m
        self.dim = dim
        self.n = 0

    def add_voter(self, embs, scores):
        """
        Add one voter

        Parameters
        _______
        embs : np.array of length dim
            The embeddings of the voter
        scoring : np.array of length m
            The scoring given by the voter
        """
        # We normalize the embedding vector
        embs = normalize(embs)
        self.embs = np.concatenate([self.embs, [embs]])
        self.scores = np.concatenate([self.scores, [scores]])
        self.n += 1

        return self

    def add_voters(self, embs, scores):
        """
        Add a group of voter

        Parameters
        _______
        embs : np.array of length dim
            The embeddings of the voter
        scoring : np.array of length m
            The scoring given by the voter
        """
        # We normalize the embedding vectors
        embs = (embs.T/np.sqrt((embs**2).sum(axis=1))).T
        self.embs = np.concatenate([self.embs, embs])
        self.scores = np.concatenate([self.scores, scores])
        self.n += len(embs)

        return self

    def add_group(self, n_voters, center, scores, dev_center=0, dev_scores=0):
        """
        Add a centered group of voter

        Parameters
        _______
        n_voters : int
            number of voters to add
        center : np.array of length dim
            embeddings of the center of the group
        scoring : np.array of length m
            average score of the group for each candidate
        dev_center : float
            deviation from the center for the embeddings
        dev_scores : float
            deviation from the average scoring vector
        """

        new_group = np.array([center] * n_voters) + np.random.normal(0, 1, (n_voters, self.dim))* dev_center
        new_group = np.maximum(0, new_group)
        new_group = np.array([normalize(x) for x in new_group])
        self.embs = np.concatenate([self.embs, new_group])

        new_scores = np.array([scores] * n_voters) + np.random.normal(0, 1, (n_voters, self.m)) * dev_scores
        new_scores = np.minimum(1, new_scores)
        new_scores = np.maximum(0, new_scores)
        self.scores = np.concatenate([self.scores, new_scores])

        self.n += n_voters

        return self

    def uniform_distribution(self, n):
        """
        Set a uniform distribution for the profile

        Parameters
        _______
        n : int
            number of voters to add
        """

        new_group = np.zeros((n, self.dim))

        for i in range(n):
            new_vec = np.abs(np.random.randn(self.dim))
            new_vec = normalize(new_vec)
            new_group[i] = new_vec

        self.embs = np.concatenate([self.embs, new_group])
        self.n += n

        new_scores = np.random.rand(n, self.m)
        self.scores = np.concatenate([self.scores, new_scores])

        return self

    def quasiorth_distribution(self, n, score_matrix, orth=0, corr=0, prob=None):

        if prob is None:
            prob = np.ones(self.dim)*1/self.dim

        new_group = np.zeros((n, self.dim))
        for i in range(n):
            r = np.random.choice(range(self.dim), p=prob)
            init_vec = np.zeros(self.dim)
            init_vec[r] = 1
            new_vec = 0.5*np.abs(np.random.randn(self.dim))
            new_vec = orth*init_vec + (1-orth)*new_vec
            new_vec = normalize(new_vec)
            new_group[i] = new_vec

        self.embs = np.concatenate([self.embs, new_group])
        self.n += n

        new_scores = corr*(new_group**2).dot(score_matrix.T) + (1-corr)*0.5*(1+np.random.randn(n, self.m))
        new_scores = np.minimum(new_scores, 1)
        new_scores = np.maximum(new_scores, 0)
        self.scores = np.concatenate([self.scores, new_scores])


        return self




    def dilate_profile_eventail(self):
        profile = self.embs

        if self.n < 2:
            raise ValueError("Cannot dilate a profile with less than 2 candidates")

        min_value = np.dot(profile[0], profile[1])
        min_index = (0, 1)
        for i in range(self.n):
            for j in range(i+1, self.n):
                val = np.dot(profile[i], profile[j])
                if val < min_value:
                    min_value = val
                    min_index = (i, j)

        (i, j) = min_index
        center = (profile[i] + profile[j])/2
        center = normalize(center)

        theta_max = np.arccos(min_value)
        k = np.pi/(2*theta_max)

        new_profile = np.zeros((self.n, self.dim))
        for i in range(self.n):
            v_i = self.embs[i]
            theta_i = np.arccos(np.dot(v_i, center))
            if theta_i == 0:
                new_profile[i] = v_i
            else:
                p_1 = np.dot(center, v_i)*center
                p_2 = v_i - p_1
                e_2 = normalize(p_2)
                new_profile[i] = center*np.cos(k*theta_i) + e_2*np.sin(k*theta_i)

        self.embs = new_profile

    def dilate_profile_parapluie(self):
        profile = self.embs

        if self.n < 2:
            raise ValueError("Cannot dilate a profile with less than 2 candidates")

        center = self.embs.sum(axis=0)
        center = center/np.linalg.norm(center)
        min_value = np.dot(profile[0], center)
        for i in range(self.n):
            val = np.dot(profile[i], center)
            if val < min_value:
                min_value = val

        theta_max = np.arccos(min_value)
        k = np.pi / (4 * theta_max)

        new_profile = np.zeros((self.n, self.dim))
        for i in range(self.n):
            v_i = self.embs[i]
            theta_i = np.arccos(np.dot(v_i, center))
            if theta_i == 0:
                new_profile[i] = v_i
            else:
                p_1 = np.dot(center, v_i) * center
                p_2 = v_i - p_1
                e_2 = normalize(p_2)
                new_profile[i] = center * np.cos(k * theta_i) + e_2 * np.sin(k * theta_i)

        self.embs = new_profile

    def dilate_profile(self, method="parapluie"):
        if method == "parapluie":
            self.dilate_profile_parapluie()
        elif method == "eventail":
            self.dilate_profile_eventail()
        else:
            raise ValueError("Incorrect method (select one among 'parpluie', 'eventail')")

    def standardization_score(self, cut_one=False):
        mu = self.scores.mean(axis=1)
        sigma = self.scores.std(axis=1)
        new_scores = self.scores.T - np.tile(mu, (self.m, 1))
        new_scores = new_scores / sigma
        new_scores += 1
        new_scores /= 2
        new_scores = np.maximum(new_scores, 0)
        if cut_one:
            new_scores = np.minimum(new_scores, 1)
        self.scores = new_scores.T

    def scored_embeddings(self, cand, rc=False):
        embeddings = []
        for i in range(self.n):
            if rc:
                s = np.sqrt(self.scores[i, cand])
            else:
                s = self.scores[i, cand]
            embeddings.append(self.embs[i] * s)
        return np.array(embeddings)

    def fake_covariance_matrix(self, cand, f, rc=False):
        matrix = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(i, self.n):
                s = self.scores[i, cand]*self.scores[j, cand]
                if rc:
                    s = np.sqrt(s)
                matrix[i, j] = f(self.embs[i], self.embs[j])*s
                matrix[j, i] = matrix[i, j]

        return matrix

    def plot_profile_3D(self, title="Profile of voters", fig=None, intfig=[1, 1, 1], show=True):
        """
        Plot the profile of voters
        """
        if fig == None:
            fig = plt.figure(figsize=(10, 10))
        ax = create_3D_plot(fig, intfig)
        for i, v in enumerate(self.embs):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=(v[0]*0.8, v[1]*0.8, v[2]*0.8), alpha=0.3)
            ax.scatter([v[0]], [v[1]], [v[2]], color='k', s=1)
        ax.set_title(title, fontsize=24)
        if show:
            plt.show()
        return ax

    def plot_profile_2D(self, title="Profile of voters", show=True):
        """
        Plot the profile of voters in 2D
        """
        tax = create_2D_plot()
        for i, v in enumerate(self.embs):
            tax.scatter([v**2], color=(v[0]*0.8, v[1]*0.8, 0), alpha=0.5, s=10)
        tax.set_title(title, fontsize=16)
        if show:
            plt.show()
        return tax

    def plot_scores_3D(self, scores, title="", fig=None, intfig=[1, 1, 1], show=True):
        """
        Plot the profile with some scoring in 3D space
        """
        if fig == None:
            fig = plt.figure(figsize=(10, 10))
        ax = create_3D_plot(fig, intfig)
        for i, (v, s) in enumerate(zip(self.embs, scores)):
            ax.plot([0, s * v[0]], [0, s * v[1]], [0, s * v[2]], color=(v[0]*0.8, v[1]*0.8, v[2]*0.8), alpha=0.3)
        ax.set_title(title, fontsize=16)
        if show:
            plt.show()
        return ax

    def plot_scores_2D(self, scores, title="", show=True):
        """
        Plot the profile with some scoring in 3D space
        """
        tax = create_2D_plot()
        for i, (v, s) in enumerate(zip(self.embs, scores)):
            tax.scatter([v**2], color=(v[0]*0.8,v[1]*0.8,0), alpha=0.5, s=s*50)
        tax.set_title(title, fontsize=16)
        if show:
            plt.show()
        return tax

    def plot_cand_3D(self, cand, fig=None, intfig=[1, 1, 1], show=True):
        """
        Plot one candidate of the election in 3D space
        """
        self.plot_scores_3D(self.scores[::, cand], title="Candidate #%i" % (cand + 1), fig=fig, intfig=intfig,
                            show=show)

    def plot_cand_2D(self, cand, show=True):
        """
        Plot one candidate of the election in 3D space
        """
        self.plot_scores_2D(self.scores[::, cand], title="Candidate #%i" % (cand + 1), show=show)

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

    def plot_cands_2D(self, list_cand=None, list_titles=None):
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
            self.plot_scores_2D(self.scores[::, cand], title=title, show=True)
            intfig[2] += 1

        plt.show()


    def copy(self, p):
        assert((self.m == p.m) and (self.dim == p.dim))
        p.profile = self.embs
        p.n = self.n
        p.scores = self.scores


class ParametriqueProfile(Profile):

    def init_profiles(self, n, score_matrix=None, prob=None):

        if prob is None:
            prob = np.ones(self.dim)

        if score_matrix is None:
            score_matrix = np.random.rand(self.m, self.dim)

        self.score_matrix = score_matrix
        self.orth_profile = np.zeros((n, self.dim))
        self.random_profile = np.zeros((n, self.dim))
        self.thetas = np.zeros(n)

        for i in range(n):
            new_vec = np.abs(np.random.randn(self.dim))
            r = np.argmax(new_vec*prob)
            new_vec = normalize(new_vec)
            self.orth_profile[i, r] = 1
            self.random_profile[i] = new_vec

            theta = np.arccos(np.dot(self.random_profile[i], self.orth_profile[i]))
            self.thetas[i] = theta

        return self

    def set_parameters(self, polarisation=0, coherence=0):

        n = len(self.thetas)
        profile = np.zeros((n, self.dim))
        for i in range(n):
            p_1 = np.dot(self.orth_profile[i], self.random_profile[i]) * self.orth_profile[i]
            p_2 = self.random_profile[i] - p_1
            e_2 = normalize(p_2)
            profile[i] = self.orth_profile[i] * np.cos(self.thetas[i]*(1-polarisation)) + e_2*np.sin(self.thetas[i]*(1-polarisation))

        self.embs = profile

        self.n = n

        new_scores = coherence*(profile**2).dot(self.score_matrix.T) + (1-coherence)*np.random.rand(n, self.m)
        new_scores = np.minimum(new_scores, 1)
        new_scores = np.maximum(new_scores, 0)
        self.scores = new_scores

        return self



