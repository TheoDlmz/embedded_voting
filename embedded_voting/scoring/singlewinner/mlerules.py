from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
import numpy as np


class MLEGaussian(ScoringRule):
    """
    A rule that computes the scores of the candidates
    with the assumption that the embeddings of the
    voters correspond to a covariance matrix.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]))
    >>> election = MLEGaussian()(ratings, embeddings)
    >>> election.scores_
    [-0.29999999999999993, 2.0, 1.4000000000000001]
    >>> election.ranking_
    [1, 2, 0]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.0, 1.0, 0.7391304347826089]
    """

    def __call__(self, ratings, embeddings=None):
        super().__call__(ratings, embeddings)
        if embeddings is None:
            raise ValueError("must specify embeddings")
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
