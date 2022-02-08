import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsRandom(EmbeddingsFromRatings):
    """
    Generates random normalized embeddings for the voters

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsRandom(2)
    >>> generator(ratings)
    Embeddings([[0.96337365, 0.26816265],
                [0.39134578, 0.92024371],
                [0.70713157, 0.70708199],
                [0.89942118, 0.43708299],
                [0.65433791, 0.75620229]])
    """
    def __init__(self, n_dim=0):
        self.n_dim = n_dim

    def __call__(self, ratings):
        ratings = Ratings(ratings)
        n_voters = ratings.shape[0]
        embs = np.abs(np.random.randn(n_voters, self.n_dim))
        return Embeddings(embs, norm=True)
