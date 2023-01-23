import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings_correlation import EmbeddingsCorrelation
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings import EmbeddingsFromRatings
from embedded_voting.utils.miscellaneous import normalize, center_and_normalize
from sklearn.decomposition import PCA


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
    def __init__(self, preprocess_ratings=None, svd_factor=0.95):
        super().__init__()
        self.svd_factor = svd_factor
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
        if self.svd_factor == "pca":
            pca_sk = PCA(n_components='mle')
            if ratings_preprocessed.shape[0] > ratings_preprocessed.shape[1]:
                pca_sk.fit(ratings_preprocessed)
            else:
                pca_sk.fit(ratings_preprocessed.T)
            n_v = pca_sk.n_components_
        else:
            if self.svd_factor == "div":
                add_div = 1
                svd_factor = 1
            else:
                add_div = 0
                svd_factor = self.svd_factor
            if s_sum == 0:
                n_v = 0
            else:
                s /= s_sum
                n_v = 0
                for s_e in s:
                    if s_e >= svd_factor*max(1 / (ratings.n_voters+add_div), 1 / (ratings.n_candidates+add_div)):
                        n_v += 1


        embeddings = EmbeddingsCorrelation(
            positions=np.dot(ratings_preprocessed, ratings_preprocessed.T),
            n_sing_val=n_v,
            ratings_means=ratings.mean(axis=1),
            ratings_stds=ratings.std(axis=1),
            norm=False
        )
        return embeddings
