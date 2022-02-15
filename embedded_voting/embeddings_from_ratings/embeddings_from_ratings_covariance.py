import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsCovariance(EmbeddingsFromRatings):
    """
    Use the covariance matrix as the embeddings for the voters.

    In this embedding, `n_dim = n_voters`, i.e. voters are embedded "in the space of voters".

    Examples
    --------
    >>> ratings = np.array([[.4, .6], [.1, .9], [.7, .5]])
    >>> embeddings_from_ratings = EmbeddingsFromRatingsCovariance()
    >>> embeddings_from_ratings(ratings)
    Embeddings([[ 0.02,  0.08, -0.02],
                [ 0.08,  0.32, -0.08],
                [-0.02, -0.08,  0.02]])
    """
    def __call__(self, ratings):
        return Embeddings(np.cov(ratings), norm=False)
