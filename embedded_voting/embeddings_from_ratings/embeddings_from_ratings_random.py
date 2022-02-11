import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings_generator_uniform import EmbeddingsGeneratorUniform
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsRandom(EmbeddingsFromRatings):
    """
    Generates random normalized embeddings for the voters.

    The embeddings of the voters are drawn uniformly at random on the part of the sphere
    where all coordinates are positive. These embeddings actually does not take
    the ratings into account.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.array([[1, 0], [1, 1], [0, 1]])
    >>> embeddings_from_ratings = EmbeddingsFromRatingsRandom(n_dim=5)
    >>> embeddings_from_ratings(ratings)
    Embeddings([[0.28396232, 0.07904315, 0.37027159, 0.87068807, 0.13386116],
                [0.12251149, 0.82631858, 0.40155802, 0.24565113, 0.28389299],
                [0.17359769, 0.1744638 , 0.09063981, 0.71672067, 0.64615953]])
    """
    def __init__(self, n_dim=0):
        self.n_dim = n_dim

    def __call__(self, ratings):
        n_voters = Ratings(ratings).shape[0]
        return EmbeddingsGeneratorUniform(n_voters=n_voters, n_dim=self.n_dim)()
