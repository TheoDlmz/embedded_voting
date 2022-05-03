import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsIdentity(EmbeddingsFromRatings):
    """
    Use the identity matrix as the embeddings for the voters.

    Intuitively, each voter is alone in her group. These embeddings actually does not take
    the ratings into account.

    Examples
    --------
    >>> ratings = np.array([[.4, .6], [.1, .9], [.7, .5]])
    >>> embeddings_from_ratings = EmbeddingsFromRatingsIdentity()
    >>> embeddings_from_ratings(ratings)
    Embeddings([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
    """
    def __call__(self, ratings):
        n_voters = Ratings(ratings).n_voters
        return Embeddings(np.eye(n_voters), norm=False)
