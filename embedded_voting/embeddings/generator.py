# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.utils.miscellaneous import normalize


class EmbeddingsGenerator:
    """
    This abstract class creates Embeddings from scratch using some function

    Parameters
    __________
    n_voters: int
        Number of voters in the embeddings
    n_dim: int
        Number of dimensions for the embeddings

    Attributes
    ----------
    n_voters: int
        Number of voters in the embeddings
    n_dim: int
        Number of dimensions for the embeddings

    """
    def __init__(self, n_voters, n_dim):
        self.n_dim = n_dim
        self.n_voters = n_voters

    def __call__(self, *args):
        """
        This function creates embeddings

        Return
        ------
        Embeddings
        """
        raise NotImplementedError


class EmbeddingsGeneratorRandom(EmbeddingsGenerator):
    """
    Create random embeddings

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = EmbeddingsGeneratorRandom(10, 2)
    >>> generator()
    Embeddings([[0.96337365, 0.26816265],
                [0.39134578, 0.92024371],
                [0.70713157, 0.70708199],
                [0.89942118, 0.43708299],
                [0.65433791, 0.75620229],
                [0.70534506, 0.70886413],
                [0.1254653 , 0.99209801],
                [0.95076   , 0.30992809],
                [0.95508537, 0.29633078],
                [0.54080587, 0.84114744]])
    """
    def __call__(self, *args):
        embs = np.abs(np.random.randn(self.n_voters, self.n_dim))
        return Embeddings(embs, norm=True)


class EmbeddingsGeneratorPolarized(EmbeddingsGenerator):
    """
    Generates parametrized embeddings with ``n_dim`` groups of voters. This class creates two embeddings: One
    according to uniform distribution, the other one with groups of voters with similar embeddings, and we can
    parametrize the embeddings to get one distribution between these two extremes.

    Parameters
    __________
    n_voters: int
        Number of voters in the embeddings
    n_dim: int
        Number of dimensions for the embeddings
    prob: list
        The probabilities for each voter to be in each group. Default is uniform distribution

    Attributes
    ----------
    n_voters: int
        Number of voters in the embeddings
    n_dim: int
        Number of dimensions for the embeddings
    prob: list
        The probabilities for each voter to be in each group. Default is uniform distribution

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = EmbeddingsGeneratorPolarized(10, 2)
    >>> generator(polarisation=1)
    Embeddings([[1., 0.],
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [0., 1.]])
    >>> generator(polarisation=0)
    Embeddings([[0.96337365, 0.26816265],
                [0.39134578, 0.92024371],
                [0.70713157, 0.70708199],
                [0.89942118, 0.43708299],
                [0.65433791, 0.75620229],
                [0.70534506, 0.70886413],
                [0.1254653 , 0.99209801],
                [0.95076   , 0.30992809],
                [0.95508537, 0.29633078],
                [0.54080587, 0.84114744]])
    >>> generator(polarisation=0.5)
    Embeddings([[0.9908011 , 0.13532618],
                [0.19969513, 0.97985808],
                [0.92388624, 0.38266724],
                [0.97453096, 0.224253  ],
                [0.34914017, 0.93707051],
                [0.38153366, 0.92435494],
                [0.06285695, 0.99802255],
                [0.98761328, 0.15690762],
                [0.98870758, 0.14985764],
                [0.28182668, 0.95946533]])

    """
    def __init__(self, n_voters, n_dim, prob=None):
        super().__init__(n_voters, n_dim)
        if prob is None:
            prob = np.ones(self.n_dim)/self.n_dim
        self.prob = list(prob)
        self._orthogonal_profile = np.zeros((n_voters, self.n_dim))
        self._random_profile = np.zeros((n_voters, self.n_dim))
        self._thetas = np.zeros(n_voters)
        self._build_profiles()

    def _build_profiles(self):
        """
        This function build the two embeddings
        of the parametrized embeddings (uniform and orthogonal).

        Return
        ------
        EmbeddingsGeneratorPolarized
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

    def __call__(self, polarisation=0.0):
        """
        Update the parameter of the parametric embeddings
        and create a new ratings.

        Parameters
        _________
        polarisation : float
            Should be between `0` and `1`.
            If it is equal to `0`, then the
            embeddings are uniformly distributed.
            If it is equal to `1`, then each voter's
            embeddings align to the dimension of its group.

        Return
        ------
        Embeddings
            The embeddings generated

        Examples
        --------
        >>> np.random.seed(42)
        >>> generator = EmbeddingsGeneratorPolarized(100, 3)
        >>> embs = generator(.8)
        >>> embs.voter_embeddings(0)
        array([0.12915167, 0.03595039, 0.99097296])
        """

        if polarisation > 1 or polarisation < 0:
            raise ValueError("Polarisation should be between 0 and 1")

        n = len(self._thetas)
        positions = np.zeros((n, self.n_dim))
        for i in range(n):
            p_1 = np.dot(self._orthogonal_profile[i], self._random_profile[i]) * self._orthogonal_profile[i]
            p_2 = self._random_profile[i] - p_1
            e_2 = normalize(p_2)
            positions[i] = self._orthogonal_profile[i] * np.cos(self._thetas[i] * (1 - polarisation)) + e_2 * np.sin(
                self._thetas[i] * (1 - polarisation))

        return Embeddings(positions)

