
from embedded_voting.scoring.multiwinner.general import IterRules, CLASSIC_QUOTA
import numpy as np


class IterSVD(IterRules):
    """
    Iterative multiwinner rule based on SVD

    Parameters
    __________
    profile : Profile
        the profile of voters
    k : int
        the size of the committee
    quota : {DROOP_QUOTA, CLASSIC_QUOTA, DROOP_QUOTA_MIN, CLASSIC_QUOTA_MIN}
        the quota used for the re-weighing step
    agg_rule : function np.array -> float
        The aggregation rule we are using
    """

    def __init__(self, profile=None, k=None, quota=CLASSIC_QUOTA, agg_rule=np.max):
        self.agg_rule = agg_rule
        super().__init__(profile=profile, k=k, quota=quota)

    def winner_k(self):
        vectors = []
        scores = []
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        for cand in range(self.profile_.m):
            X = np.dot(np.diag(self.weights), self.profile_.scored_embeddings(cand))
            _, s, v = np.linalg.svd(X, full_matrices=False)
            scores.append(self.agg_rule(s))
            if (v[0] <= 0).all():
                v[0] = -v[0]
            vectors.append(v[0])
        return scores, vectors

    def satisfaction(self, cand, vec):
        return [self.profile_.scores[i, cand] * np.dot(self.profile_.embs[i], vec) for i in range(self.profile_.n)]
