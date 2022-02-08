import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsIdentity(EmbeddingsFromRatings):
    """
    Use the identity matrix as the embeddings for the voters

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsIdentity()
    >>> generator(ratings)
    Embeddings([[1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.]])
    """
    def __call__(self, ratings):
        ratings = Ratings(ratings)
        n_dim = ratings.shape[0]
        return Embeddings(np.eye(n_dim))
