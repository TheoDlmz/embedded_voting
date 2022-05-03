import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings_correlation import EmbeddingsCorrelation
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings
from embedded_voting.utils.miscellaneous import normalize, center_and_normalize


class EmbeddingsFromRatingsCorrelation(EmbeddingsFromRatings):
    """
    Use the correlation with each voter as the embeddings.

    Morally, we have two levels of embedding.

    * First, `v_i = preprocess_ratings(ratings_voter_i)` for each voter `i`, which is used as a
      computation step but not recorded.
    * Second, `M = v @ v.T`, which is recorded as the final embeddings.

    Other attributes are computed and recorded:

    * `n_sing_val`: the number of relevant singular values when we compute the SVD.
       This is based on the Principal Component Analysis (PCA).
    * `ratings_means`: the mean rating for each voter (without preprocessing).
    * `ratings_stds`: the standard deviation of the ratings for each voter (without preprocessing).

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings = np.ones((5, 3))
    >>> generator = EmbeddingsFromRatingsCorrelation(preprocess_ratings=normalize)
    >>> embeddings = generator(ratings)
    >>> embeddings
    EmbeddingsCorrelation([[1., 1., 1., 1., 1.],
                           [1., 1., 1., 1., 1.],
                           [1., 1., 1., 1., 1.],
                           [1., 1., 1., 1., 1.],
                           [1., 1., 1., 1., 1.]])
    >>> embeddings.n_sing_val
    1

    In fact, the typical usage is with `center_and_normalize`:

    >>> generator = EmbeddingsFromRatingsCorrelation(preprocess_ratings=center_and_normalize)
    >>> embeddings = generator(ratings)
    >>> embeddings
    EmbeddingsCorrelation([[0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]])
    >>> embeddings.n_sing_val
    0
    """
    def __init__(self, preprocess_ratings=None):
        super().__init__()
        self.preprocess_ratings = preprocess_ratings

    def __call__(self, ratings):
        ratings = Ratings(ratings)
        if self.preprocess_ratings is None:
            ratings_preprocessed = ratings
        else:
            ratings_preprocessed = Ratings([self.preprocess_ratings(ratings_voter) for ratings_voter in ratings])

        u, s, v = np.linalg.svd(ratings_preprocessed)
        s = np.sqrt(s)
        s_sum = s.sum()
        if s_sum == 0:
            n_v = 0
        else:
            s /= s_sum
            n_v = 0
            for s_e in s:
                if s_e >= max(1 / ratings.n_voters, 1 / ratings.n_candidates):
                    n_v += 1

        embeddings = EmbeddingsCorrelation(
            positions=np.dot(ratings_preprocessed, ratings_preprocessed.T),
            n_sing_val=n_v,
            ratings_means=ratings.mean(axis=1),
            ratings_stds=ratings.std(axis=1),
            norm=False
        )
        return embeddings
