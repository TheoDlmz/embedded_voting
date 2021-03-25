from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile
import numpy as np


class MLERule(ScoringRule):
    """
    A rule that computes the scores of the candidates
    with the assumption that the embeddings of the
    voters correspond to a covariance matrix.

    Examples
    --------
    >>> my_profile = Profile(3, 3)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = np.cov([[1, 1], [1, 0], [0, 1]])
    >>> _ = my_profile.add_voters(embeddings, scores, normalize_embs=False)
    >>> election = MLERule(my_profile)
    >>> election.scores_
    [0.30000000000000004, 2.0, 1.4000000000000001]
    >>> election.ranking_
    [1, 2, 0]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.0, 1.0, 0.6470588235294118]
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


