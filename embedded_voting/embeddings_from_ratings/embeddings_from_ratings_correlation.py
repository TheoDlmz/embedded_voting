import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings


class EmbeddingsFromRatingsCorrelation(EmbeddingsFromRatings):
    """
    Use the correlation with each voter as the embeddings

    Attributes
    ----------
    n_sing_val_: int
        The number of relevant singular values when we compute the SVD. based on the Principal Component
        Analysis (PCA)

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsCorrelation()
    >>> embeddings = generator(ratings)
    >>> embeddings
    Embeddings([[0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136],
                [0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136]])
    >>> embeddings.n_sing_val_
    1

    """
    def __init__(self):
        super().__init__()
        self.n_sing_val_ = None

    def __call__(self, ratings):
        ratings = Ratings(ratings)
        positions = (ratings.T / np.sqrt((ratings ** 2).sum(axis=1))).T
        n_voters, n_candidates = ratings.shape
        self.n_dim = n_candidates

        u, s, v = np.linalg.svd(positions)

        n_voters, n_candidates = positions.shape
        s = np.sqrt(s)
        s /= s.sum()
        n_v = 0
        for s_e in s:
            if s_e >= max(1 / n_voters, 1 / n_candidates):
                n_v += 1

        embeddings = Embeddings(np.dot(positions, positions.T), norm=True)
        embeddings.n_sing_val_ = n_v
        return embeddings
