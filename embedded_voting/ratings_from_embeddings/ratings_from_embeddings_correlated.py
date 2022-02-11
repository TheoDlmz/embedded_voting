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
    scores_matrix: np.ndarray or list
        An array with shape ``n_dim, n_candidates`` such that ``scores_matrix[i,j]`` determines the rating
        given by the group of voter in the dimension i to candidate j. If none is specified, a random one
        is generated

    Examples
    --------
    >>> np.random.seed(42)
    >>> embeddings = Embeddings(np.array([[0, 1], [1, 0], [1, 1]]), norm=True)
    >>> generator = RatingsFromEmbeddingsCorrelated(2, 3, .5, scores_matrix=np.array([[.8,.4],[.1,.7]]))
    >>> generator(embeddings)
    Ratings([[0.23727006, 0.82535715],
             [0.76599697, 0.49932924],
             [0.30300932, 0.35299726]])
    """

    def __init__(self, n_candidates, n_dim, coherence=0, scores_matrix=None):
        super().__init__(n_candidates)
        self.n_dim = n_dim
        self.coherence = coherence
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.scores_matrix = np.array(scores_matrix)

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
        ratings = self.coherence * (positions ** 2).dot(self.scores_matrix)
        ratings += (1 - self.coherence) * np.random.rand(n_voters, self.n_candidates)
        ratings = np.minimum(ratings, 1)
        ratings = np.maximum(ratings, 0)
        return Ratings(ratings)

    def set_scores(self, scores_matrix=None):
        """
        Update the scores matrix of
        the parametric ratings.

        Parameters
        ----------
        scores_matrix : np.ndarray or list
            Matrix of shape :attr:`n_dim`, :attr:`n_candidates` containing the scores given by
            each group. More precisely, `scores_matrix[i,j]` is the score given by the group
            represented by the dimension i to the candidate j.
            By default, it is set at random with a uniform distribution.

        Return
        ------
        ProfileGenerator
            The object itself.

        Examples
        --------
        >>> generator = RatingsFromEmbeddingsCorrelated(2, 3)
        >>> generator.set_scores(np.array([[.8,.4],[.1,.7]]))
        <embedded_voting.ratings_from_embeddings.ratings_from_embeddings_correlated.RatingsFromEmbeddingsCorrelated object at ...>
        >>> generator.scores_matrix
        array([[0.8, 0.4],
               [0.1, 0.7]])
        """
        if scores_matrix is None:
            scores_matrix = np.random.rand(self.n_dim, self.n_candidates)
        self.scores_matrix = np.array(scores_matrix)
        return self
