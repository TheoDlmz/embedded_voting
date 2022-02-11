import numpy as np

from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings_from_embeddings import RatingsFromEmbeddings


class RatingsFromEmbeddingsCorrelated(RatingsFromEmbeddings):
    """
    This method create ratings correlated to the embeddings by a score matrix.

    Parameters
    ----------
    n_candidates: int
        The number of candidates wanted in the ratings
    n_dim: int
        The number of dimension of the embeddings
    coherence: float
        Between 0 and 1, indicates the desired level of correlation between embeddings and ratings. If 0,
        ratings are random, if 1, ratings are perfectly correlated to embeddings.
    ratings_dim_candidate: np.ndarray or list
        An array with shape ``n_dim, n_candidates`` such that ``ratings_dim_candidate[i,j]`` determines the rating
        given by the group of voter in the dimension i to candidate j. If none is specified, a random one
        is generated
    ratings_dim_candidate : np.ndarray or list
        Matrix of shape :attr:`n_dim`, :attr:`n_candidates` containing the scores given by
        each group. More precisely, `ratings_dim_candidate[i,j]` is the score given by the group
        represented by the dimension i to the candidate j.
        By default, it is set at random with a uniform distribution.

    Examples
    --------
    >>> np.random.seed(42)
    >>> embeddings = Embeddings(np.array([[0, 1], [1, 0], [1, 1]]), norm=True)
    >>> generator = RatingsFromEmbeddingsCorrelated(n_candidates=2, n_dim=3, coherence=.5, ratings_dim_candidate=np.array([[.8,.4],[.1,.7]]))
    >>> generator(embeddings)
    Ratings([[0.23727006, 0.82535715],
             [0.76599697, 0.49932924],
             [0.30300932, 0.35299726]])
    """

    def __init__(self, coherence=0, ratings_dim_candidate=None, n_candidates=None, n_dim=None):
        super().__init__(n_candidates)
        self.n_dim = n_dim
        self.coherence = coherence
        if ratings_dim_candidate is None:
            ratings_dim_candidate = np.random.rand(self.n_dim, self.n_candidates)
        self.ratings_dim_candidate = np.array(ratings_dim_candidate)

    def __call__(self, embeddings, *args):
        """
        This method generate ratings from the embeddings using the score matrix.

        Parameters
        ----------
        embeddings: Embeddings
            The embeddings we want to use to obtain the ratings

        Return
        ------
        Ratings
        """
        embeddings = Embeddings(embeddings, norm=True)
        positions = np.array(embeddings)
        n_voters = embeddings.n_voters
        ratings = self.coherence * (positions ** 2).dot(self.ratings_dim_candidate)
        ratings += (1 - self.coherence) * np.random.rand(n_voters, self.n_candidates)
        ratings = np.minimum(ratings, 1)
        ratings = np.maximum(ratings, 0)
        return Ratings(ratings)
