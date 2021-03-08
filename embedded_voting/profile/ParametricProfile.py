# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.profile.Profile import Profile


class ParametricProfile(Profile):
    """
    A parametric profile of voter. It creates two profiles : one orthogonal and one uniformly distributed.

    Parameters
    _________
    n_candidates : int
        The number of candidate for this profile
    n_dim : int
        The number of dimensions for the voters' embeddings
    n_voters : int
        The number of voter of the profile
    scores_matrix : n_dim x n_candidates matrix of float
        scores_matrix[i,j] is the score given by the group represented by the dimension i
        to the candidate j. Default is random with uniform distribution.
    prob : array of length n_dim
        prob[i] is the probability for a voter to be on the group represented by the dimension i.
        default is uniform distribution.
    """

    def __init__(self, n_candidates, n_dim, n_voters, scores_matrix=None, prob=None):
        super().__init__(n_candidates, n_dim)

        if prob is None:
            prob = np.ones(self.n_dim)

        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_candidates, self.n_dim)

        self.score_matrix = scores_matrix
        self.orthogonal_profile = np.zeros((n_voters, self.n_dim))
        self.random_profile = np.zeros((n_voters, self.n_dim))
        self.thetas = np.zeros(n_voters)

        for i in range(n_voters):
            new_vec = np.abs(np.random.randn(self.n_dim))
            r = np.argmax(new_vec * prob)
            new_vec = normalize(new_vec)
            self.orthogonal_profile[i, r] = 1
            self.random_profile[i] = new_vec

            theta = np.arccos(np.dot(self.random_profile[i], self.orthogonal_profile[i]))
            self.thetas[i] = theta

    def set_parameters(self, polarisation=0, coherence=0):
        """
        Set the parameters of the parametric profile to create a real profile

        Parameters
        _________
        polarisation : float between 0 and 1.
            If it is equal to 0, then the embeddings are uniformly distributed. If it is equal to 1,
            then each voter's embeddings align to the dimension of its group.
        coherence : float between 0 and 1.
            If it is equal to 0, the scores are randomly drawn and does not depends on voters' embeddings.
            If it is equal to 1, the scores given by a voter only depend on its embeddings.
        """
        n = len(self.thetas)
        profile = np.zeros((n, self.n_dim))
        for i in range(n):
            p_1 = np.dot(self.orthogonal_profile[i], self.random_profile[i]) * self.orthogonal_profile[i]
            p_2 = self.random_profile[i] - p_1
            e_2 = normalize(p_2)
            profile[i] = self.orthogonal_profile[i] * np.cos(self.thetas[i] * (1 - polarisation)) + e_2 * np.sin(
                self.thetas[i] * (1 - polarisation))

        self.embeddings = profile
        self.n_voters = n

        new_scores = coherence * (profile ** 2).dot(self.score_matrix.T) \
            + (1 - coherence) * np.random.rand(n, self.n_candidates)
        new_scores = np.minimum(new_scores, 1)
        new_scores = np.maximum(new_scores, 0)
        self.scores = new_scores
        return self
