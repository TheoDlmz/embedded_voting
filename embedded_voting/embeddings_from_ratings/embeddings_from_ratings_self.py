import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsSelf(EmbeddingsFromRatings):
    """
    Use the normalized ratings as the embeddings for the voters.

    Parameters
    ----------
    norm: bool
        Whether the embeddings should be normalized.

    Examples
    --------
    >>> ratings = np.array([[1, 0], [1, 1], [0, 1]])
    >>> embeddings_from_ratings = EmbeddingsFromRatingsSelf(norm=True)
    >>> embeddings_from_ratings(ratings)
    Embeddings([[1.        , 0.        ],
                [0.70710678, 0.70710678],
                [0.        , 1.        ]])
    """
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, ratings):
        return Embeddings(ratings, norm=self.norm)
