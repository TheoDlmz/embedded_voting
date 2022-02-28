import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings.embeddings_generator import EmbeddingsGenerator
from embedded_voting.embeddings.embeddings_generator_uniform import EmbeddingsGeneratorUniform
from embedded_voting.embeddings.embeddings_generator_fully_polarized import EmbeddingsGeneratorFullyPolarized


class EmbeddingsGeneratorPolarized(EmbeddingsGenerator):
    """
    Generates parametrized embeddings with ``n_dim`` groups of voters. This class creates two embeddings: one
    according to uniform distribution, the other one fully polarized (with groups of voters on the canonical basis),
    and we can parametrize the embeddings to get one distribution between these two extremes.

    Parameters
    __________
    n_voters: int
        Number of voters in the embeddings.
    n_dim: int
        Number of dimensions for the embeddings.
    prob: list
        The probabilities for each voter to be in each group. Default is uniform distribution.

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = EmbeddingsGeneratorPolarized(10, 2)
    >>> generator(polarisation=1)
    Embeddings([[1., 0.],
                [0., 1.],
                [1., 0.],
                [0., 1.],
                [0., 1.],
                [1., 0.],
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
                [0.53052663, 0.84766827],
                [0.34914017, 0.93707051],
                [0.92340269, 0.38383261],
                [0.06285695, 0.99802255],
                [0.98761328, 0.15690762],
                [0.98870758, 0.14985764],
                [0.28182668, 0.95946533]])
    """
    def __init__(self, n_voters, n_dim, prob=None):
        super().__init__(n_voters, n_dim)
        if prob is None:
            prob = np.ones(self.n_dim)
        self.prob = np.array(prob) / np.sum(prob)
        # Two basic embeddings
        self._random_profile = EmbeddingsGeneratorUniform(n_voters=self.n_voters, n_dim=self.n_dim)()
        self._fully_polarized_profile = EmbeddingsGeneratorFullyPolarized(
            n_voters=self.n_voters, n_dim=n_dim, prob=prob)()

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
        >>> embeddings = generator(.8)
        >>> embeddings.voter_embeddings(0)
        array([0.9828518 , 0.03849652, 0.18033401])
        """
        return self._fully_polarized_profile.mixed_with(self._random_profile, intensity=1 - polarisation)
