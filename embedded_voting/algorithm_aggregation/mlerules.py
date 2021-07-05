from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile
import numpy as np


class MLEGaussian(ScoringRule):
    """
    A rule that computes the scores of the candidates
    with the assumption that the embeddings of the
    voters correspond to a covariance matrix.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores, normalize_embs=False)
    >>> election = MLEGaussian(my_profile)
    >>> election.scores_
    [0.30000000000000004, 2.0, 1.4000000000000001]
    >>> election.ranking_
    [1, 2, 0]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.0, 1.0, 0.6470588235294118]
    """
    def __init__(self,  profile=None):
        super().__init__(profile=profile)
        if profile is not None:
            self(profile)

    def __call__(self, profile):
        self.inverse_cov = np.linalg.pinv(np.cov(self.profile_.embeddings)).sum(axis=0)
        return self

    def score_(self, candidate):
        scores = self.profile_.scores[::, candidate]
        sum_cov = self.inverse_cov
        score = 0
        for i in range(len(scores)):
            score += scores[i]*sum_cov[i]
        return score/sum_cov.sum()


