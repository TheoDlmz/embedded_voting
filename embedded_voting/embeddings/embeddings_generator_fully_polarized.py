import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings.embeddings_generator import EmbeddingsGenerator


class EmbeddingsGeneratorFullyPolarized(EmbeddingsGenerator):
    """
    Create embeddings that are random vectors of the canonical basis.

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
    >>> generator = EmbeddingsGeneratorFullyPolarized(10, 5)
    >>> generator()
    Embeddings([[0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 1., 0.],
                [0., 0., 1., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 1., 0.]])
    """
    def __init__(self, n_voters, n_dim, prob=None):
        super().__init__(n_voters, n_dim)
        if prob is None:
            prob = np.ones(self.n_dim)
        self.prob = np.array(prob) / np.sum(prob)

    def __call__(self, *args):
        voter_groups = np.random.choice(self.n_dim, self.n_voters, p=self.prob)
        return Embeddings(np.eye(self.n_dim)[voter_groups, :], norm=False)
