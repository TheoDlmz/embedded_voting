import numpy as np
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_self import EmbeddingsFromRatingsSelf
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_covariance import EmbeddingsFromRatingsCovariance
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property


class RuleMLEGaussian(Rule):
    """
    A rule that computes the scores of the candidates, assuming that the embeddings of the voters correspond to a
    covariance matrix.

    For this rule, the embeddings must be a matrix `n_voters` * `n_voters`.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> election = RuleMLEGaussian()(ratings)
    >>> election.scores_
    [0.507..., 0.606..., 0.275...]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.70..., 1.0, 0.0]

    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = EmbeddingsFromRatingsCovariance()(ratings)
    >>> election = RuleMLEGaussian()(ratings, embeddings)
    >>> election.scores_
    [0.507..., 0.606..., 0.275...]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.70..., 1.0, 0.0]
    """

    def __init__(self, embeddings_from_ratings=None):
        if embeddings_from_ratings is None:
            embeddings_from_ratings = EmbeddingsFromRatingsCovariance()
        super().__init__(score_components=1, embeddings_from_ratings=embeddings_from_ratings)

    @cached_property
    def pinv_covariance_(self):
        return np.array(np.linalg.pinv(self.embeddings_))

    @cached_property
    def weights_(self):
        return self.pinv_covariance_.sum(axis=0)

    @cached_property
    def weights_normalized_(self):
        return self.weights_ / self.weights_.sum()

    def _score_(self, candidate):
        return self.ratings_.candidate_ratings(candidate) @ self.weights_normalized_
