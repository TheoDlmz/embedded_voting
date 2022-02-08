import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsSelf(EmbeddingsFromRatings):
    """
    Use the normalized ratings as the embeddings for the voters

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsSelf()
    >>> generator(ratings)
    Embeddings([[0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027],
                [0.57735027, 0.57735027, 0.57735027]])
    """
    def __call__(self, ratings):
        return Embeddings(ratings, norm=True)
