from embedded_voting.scoring.singlewinner.general import ScoringRule
import numpy as np


class MLERule(ScoringRule):
    """
    A rule that computes the scores of the candidates
    with the assumption that the embeddings of the
    voters correspond to a covariance matrix.
    """

    def score_(self, candidate):
        cov = self.profile_.embeddings
        scores = self.profile_.scores[::, candidate]
        inverse_cov = np.linalg.pinv(cov)
        sum_cov = inverse_cov.sum(axis=0)
        score = 0
        for i in range(len(scores)):
            score += scores[i]*sum_cov[i]
        return score/sum_cov.sum()


