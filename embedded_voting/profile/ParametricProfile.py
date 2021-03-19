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
    A parametric profile of voter with embeddings.
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
    >>> my_profile = ParametricProfile(4, 3, 100)
    >>> my_profile.n_voters
    100
    >>> my_profile.n_dim
    3
    >>> my_profile.n_candidates
    4
    >>> len(my_profile._thetas)
    100

    """

    def __init__(self, n_candidates, n_dim, n_voters, scores_matrix=None, prob=None):
        super().__init__(n_candidates, n_dim)
        if prob is None:
            prob = np.ones(self.n_dim)/self.n_dim
        self.prob = list(prob)
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.n_dim = n_dim
        self.n_candidates = n_candidates
        self.n_voters = n_voters
        self.scores_matrix = np.array(scores_matrix).T
        self._orthogonal_profile = np.zeros((n_voters, self.n_dim))
        self._random_profile = np.zeros((n_voters, self.n_dim))
        self._thetas = np.zeros(n_voters)

        self._build_profiles()

    def _build_profiles(self, set_parameters=True):
        """
        This function build the two profiles
        of the parametric profile (uniform and orthogonal).

        Parameters
        ----------
        set_parameters:
            If True, set both parameters
            to 0 at the end of the function.

        Return
        ------
        ParametricProfile
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

        if set_parameters:
            self.set_parameters()
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
        ParametricProfile
            The object itself.

        Examples
        --------
        >>> my_profile = ParametricProfile(4, 3, 100)
        >>> my_profile.set_scores(np.ones((3, 4)))
        >>> my_profile.scores_matrix
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

        self._build_profiles(False)

    def set_parameters(self, polarisation=0.0, coherence=0.0):
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
            the profile itself

        Examples
        --------
        >>> my_profile = ParametricProfile(4, 3, 100)
        >>> my_profile.set_parameters(.8, .8)
        <embedded_voting.profile.ParametricProfile.ParametricProfile object at ...>
        """

        if polarisation > 1 or coherence > 1:
            raise ValueError("Parameters should be between 0 and 1")

        n = len(self._thetas)
        profile = np.zeros((n, self.n_dim))
        for i in range(n):
            p_1 = np.dot(self._orthogonal_profile[i], self._random_profile[i]) * self._orthogonal_profile[i]
            p_2 = self._random_profile[i] - p_1
            e_2 = normalize(p_2)
            profile[i] = self._orthogonal_profile[i] * np.cos(self._thetas[i] * (1 - polarisation)) + e_2 * np.sin(
                self._thetas[i] * (1 - polarisation))

        self.embeddings = profile
        self.n_voters = n

        new_scores = coherence * (profile ** 2).dot(self.scores_matrix.T) \
            + (1 - coherence) * np.random.rand(n, self.n_candidates)
        new_scores = np.minimum(new_scores, 1)
        new_scores = np.maximum(new_scores, 0)
        self.scores = new_scores
        return self
