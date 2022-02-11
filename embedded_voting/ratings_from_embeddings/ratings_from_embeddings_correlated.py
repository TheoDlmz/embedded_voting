import numpy as np

from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings_from_embeddings import RatingsFromEmbeddings
from embedded_voting.ratings.ratings_generator_uniform import RatingsGeneratorUniform


class RatingsFromEmbeddingsCorrelated(RatingsFromEmbeddings):
    """
    Generate ratings from embeddings and from a matrix where each embedding dimension gives a rating to each candidate.

    `ratings_automatic[voter, candidate]` is computed as the average of `ratings_dim_candidate[:, candidate]`, weighted
    by the squares of `emdeddings[voter, :]`. In particular, for each `voter` belonging to group `i` (in the sense that
    their embedding is the i-th vector of the canonical basis), then `ratings_automatic[voter, candidate]` is equal to
    `ratings_dim_candidate[i, candidate]`.

    `ratings_random[voter, candidate]` is computed as a uniform random number between `minimum_random_rating` and
    `maximum_random_rating`.

    Finally, `ratings` is the barycenter: `coherence * ratings_automatic + (1 - coherence) * ratings_random`.

    Parameters
    ----------
    coherence: float
        Between 0 and 1, indicates the degree of coherence between voters having similar embeddings.
        If 0, the ratings are purely random. If 1, the ratings are automatically deduced
        from `embeddings` and `ratings_dim_candidate`.
    ratings_dim_candidate: np.ndarray or list
        An array with shape :attr:`n_dim`, :attr:`n_candidates`. The coefficient
        `ratings_dim_candidate[dim, candidate]` is the score given by the group represented by the dimension `dim`
        to the `candidate`. By default, it is set at random with a uniform distribution in the interval
        [`minimum_random_rating`, `maximum_random_rating`].
    n_dim: int
        The number of dimension of the embeddings. Used to generate `ratings_dim_candidate` if it is not specified.
    n_candidates: int
        The number of candidates. Used to generate `ratings_dim_candidate` if it is not specified.
    minimum_random_rating: float
        Minimum rating for the random part.
    maximum_random_rating: float
        Maximum rating for the random part.
    clip: bool
        If true, the final ratings are clipped in the interval [`minimum_random_rating`, `maximum_random_rating`].

    Examples
    --------
    >>> np.random.seed(42)
    >>> embeddings = Embeddings(np.array([[0, 1], [1, 0], [1, 1]]), norm=True)
    >>> generator = RatingsFromEmbeddingsCorrelated(coherence=.5, ratings_dim_candidate=np.array([[.8,.4],[.1,.7]]))
    >>> generator(embeddings)
    Ratings([[0.23727006, 0.82535715],
             [0.76599697, 0.49932924],
             [0.30300932, 0.35299726]])
    """

    def __init__(self, coherence=0, ratings_dim_candidate=None, n_dim=None, n_candidates=None,
                 minimum_random_rating=0, maximum_random_rating=1, clip=False):
        if ratings_dim_candidate is None:
            ratings_dim_candidate = (
                np.random.rand(n_dim, n_candidates) * (maximum_random_rating - minimum_random_rating)
                + minimum_random_rating
            )
        else:
            if n_dim is not None and n_dim != ratings_dim_candidate.shape[0]:
                raise ValueError("n_dim should be omitted or equal to ratings_dim_candidate.shape[0].")
            if n_candidates is not None and n_candidates != ratings_dim_candidate.shape[1]:
                raise ValueError("n_candidates should be omitted or equal to ratings_dim_candidate.shape[1].")
            ratings_dim_candidate = np.array(ratings_dim_candidate)
            n_dim, n_candidates = ratings_dim_candidate.shape
        # Store variables
        self.coherence = coherence
        self.ratings_dim_candidate = ratings_dim_candidate
        self.n_dim = n_dim
        self.minimum_random_rating = minimum_random_rating
        self.maximum_random_rating = maximum_random_rating
        self.clip = clip
        super().__init__(n_candidates)

    def __call__(self, embeddings, *args):
        """
        This method generate ratings from the embeddings, possibly with a random component.

        Parameters
        ----------
        embeddings : Embeddings

        Return
        ------
        ratings : Ratings
        """
        embeddings = Embeddings(embeddings, norm=True)
        ratings_automatic = embeddings ** 2 @ self.ratings_dim_candidate
        ratings_random = RatingsGeneratorUniform(
            n_voters=embeddings.n_voters,
            minimum_rating=self.minimum_random_rating,
            maximum_rating=self.maximum_random_rating
        )(self.n_candidates)
        ratings = self.coherence * ratings_automatic + (1 - self.coherence) * ratings_random
        if self.clip:
            ratings = np.clip(ratings, self.minimum_random_rating, self.maximum_random_rating)
        return Ratings(ratings)
