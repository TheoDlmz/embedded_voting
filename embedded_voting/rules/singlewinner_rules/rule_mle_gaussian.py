from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_identity import EmbeddingsFromRatingsIdentity
import numpy as np


class RuleMLEGaussian(Rule):
    """
    A rule that computes the scores of the candidates
    with the assumption that the embeddings of the
    voters correspond to a covariance matrix.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> election = RuleMLEGaussian(embeddings_from_ratings=EmbeddingsFromRatingsIdentity())(ratings)
    >>> election.scores_
    [0.507..., 0.606..., 0.275...]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.70..., 1.0, 0.0]
    """

    def __call__(self, ratings, embeddings=None):
        super().__call__(ratings, embeddings)
        if embeddings is None:
            positions = np.array(ratings)
        else:
            positions = np.array(self.embeddings_)
        self.inverse_cov = np.linalg.pinv(np.cov(positions)).sum(axis=0)
        return self

    def _score_(self, candidate):
        scores = self.ratings_.candidate_ratings(candidate)
        sum_cov = self.inverse_cov
        score = 0
        for i in range(len(scores)):
            score += scores[i]*sum_cov[i]
        return score/sum_cov.sum()
