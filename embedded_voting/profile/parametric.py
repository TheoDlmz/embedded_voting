# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.profile.profile import Profile
from embedded_voting.embeddings.embeddings import Embeddings


class ProfileGenerator:
    """
    A parametric profile generator of voter with embeddings.
    It creates two profiles : one with orthogonal embeddings
    and preferences correlated to embeddings
    and another one with random embeddings and random scores.
    You can then creates profiles between these
    two extremes by playing with the parameters.

    Parameters
    _________
    n_candidates : int
        The number of candidates in the profile.
    n_dim : int
        The number of dimensions of the voters' embeddings.
    n_voters : int
        The number of voters in the profile.
    scores_matrix : np.ndarray or list
        Matrix of shape :attr:`n_dim`, :attr:`n_candidates`
        containing the scores given by each group.
        More precisely, `scores_matrix[i,j]` is
        the score given by the group
        represented by the dimension i
        to the candidate j. By default,
        it is set at random with a
        uniform distribution.
    prob : np.ndarray or list
        `prob[i]` is the probability for a voter
        to be on the group represented by the dimension i.
        Should be of length :attr:`n_dim`.
        By default, it is the set at 1/:attr:`n_dim`
        for every dimensions.

    Attributes
    ----------
    n_candidates : int
        The number of candidates in the profile.
    n_dim : int
        The number of dimensions of the voters' embeddings.
    n_voters : int
        The number of voters in the profile.
    scores_matrix : np.ndarray
        Matrix of shape :attr:`n_dim`, :attr:`n_candidates`
        containing the scores given by each group.
        More precisely, `scores_matrix[i,j]` is
        the score given by the group
        represented by the dimension i
        to the candidate j.
    prob : list
        `prob[i]` is the probability for a
        voter to be on the group represented by the dimension `i`.
        Should be of length :attr:`n_dim`.

    Examples
    --------
    >>> my_profile = ProfileGenerator(100, 4, 3)
    >>> my_profile.n_voters
    100
    >>> my_profile.n_dim
    3
    >>> my_profile.n_candidates
    4

    """

    def __init__(self, n_voters, n_candidates, n_dim=3, scores_matrix=None, prob=None):
        """
        _orthogonal_profile : np.ndarray
            matrix of shape :attr:`n_voters`, :attr:`n_dim`
            containing the "orthogonal" profile.
        _random_profile : np.ndarray
            matrix of shape :attr:`n_voters`, :attr:`n_dim`
            containing the "random" profile.
        _thetas : list
            list of length :attr:`n_voters` containing
            angular distances between the embeddings of the two
            profiles for each voter.
        """

        self.n_dim = n_dim
        self.n_candidates = n_candidates
        self.n_voters = n_voters
        if prob is None:
            prob = np.ones(self.n_dim)/self.n_dim
        self.prob = list(prob)
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.scores_matrix = np.array(scores_matrix).T
        self._orthogonal_profile = np.zeros((n_voters, self.n_dim))
        self._random_profile = np.zeros((n_voters, self.n_dim))
        self._thetas = np.zeros(n_voters)

        self._build_profiles()

    def _build_profiles(self):
        """
        This function build the two profiles
        of the parametric profile (uniform and orthogonal).


        Return
        ------
        ProfileGenerator
            The object itself.
        """

        for i in range(self.n_voters):
            new_vec = np.abs(np.random.randn(self.n_dim))
            r = np.argmax(new_vec * self.prob)
            new_vec = normalize(new_vec)
            self._orthogonal_profile[i, r] = 1
            self._random_profile[i] = new_vec

            theta = np.arccos(np.dot(self._random_profile[i], self._orthogonal_profile[i]))
            self._thetas[i] = theta

        return self

    def set_scores(self, scores_matrix=None):
        """
        Update the scores matrix of
        the parametric profile.

        Parameters
        ----------
        scores_matrix : np.ndarray or list
            Matrix of shape :attr:`n_dim`, :attr:`n_candidates` containing the scores given by
            each group. More precisely, `scores_matrix[i,j]` is the score given by the group
            represented by the dimension i to the candidate j.
            By default, it is set at random with a uniform distribution.

        Return
        ------
        ProfileGenerator
            The object itself.

        Examples
        --------
        >>> generator = ProfileGenerator(100, 4, 3)
        >>> generator.set_scores(np.ones((3, 4)))
        >>> generator.scores_matrix
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])
        """
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.scores_matrix = np.array(scores_matrix).T
        self._orthogonal_profile = np.zeros((self.n_voters, self.n_dim))
        self._random_profile = np.zeros((self.n_voters, self.n_dim))
        self._thetas = np.zeros(self.n_voters)

        self._build_profiles()

    def __call__(self, polarisation=0.0, coherence=0.0):
        """
        Update the parameters of the parametric profile
        and create a new profile.

        Parameters
        _________
        polarisation : float
            Should be between `0` and `1`.
            If it is equal to `0`, then the
            embeddings are uniformly distributed.
            If it is equal to `1`, then each voter's
            embeddings align to the dimension of its group.
        coherence : float
            Should be between `0` and `1`.
            If it is equal to `0`, the scores are
            randomly drawn and do not
            depends on voters' embeddings.
            If it is equal to `1`, the scores
            given by a voter only depend on his embeddings.

        Return
        ------
        Profile
            The profile generated

        Examples
        --------
        >>> np.random.seed(42)
        >>> generator = ProfileGenerator(4, 3, 100)
        >>> profile = generator(.8, .8)
        >>> profile.ratings
        array([[0.12464552, 0.62869707, 0.75700584],
               [0.30273919, 0.81757126, 0.81867831],
               [0.40513352, 0.93730291, 0.81315598],
               [0.32305075, 0.25390238, 0.62966269]])
        """

        if polarisation > 1 or coherence > 1 or polarisation < 0 or coherence < 0:
            raise ValueError("Parameters should be between 0 and 1")

        n = len(self._thetas)
        positions = np.zeros((n, self.n_dim))
        for i in range(n):
            p_1 = np.dot(self._orthogonal_profile[i], self._random_profile[i]) * self._orthogonal_profile[i]
            p_2 = self._random_profile[i] - p_1
            e_2 = normalize(p_2)
            positions[i] = self._orthogonal_profile[i] * np.cos(self._thetas[i] * (1 - polarisation)) + e_2 * np.sin(
                self._thetas[i] * (1 - polarisation))

        ratings = coherence * (positions ** 2).dot(self.scores_matrix.T) \
            + (1 - coherence) * np.random.rand(n, self.n_candidates)
        ratings = np.minimum(ratings, 1)
        ratings = np.maximum(ratings, 0)
        embeddings = Embeddings(positions)
        return Profile(ratings, embeddings)
