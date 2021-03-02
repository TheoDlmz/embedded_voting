
from embedded_voting.scoring.multiwinner.general import IterRules, CLASSIC_QUOTA
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
import numpy as np


class IterFeatures(IterRules):
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
    """

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self):
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        features = self.compute_features(self.profile_.embs,
                                         np.dot(np.diag(self.weights), self.profile_.scores))
        scores = np.sum(features ** 2, axis=1)
        return scores, features

    def satisfaction(self, winner_j, vec):
        temp = [np.dot(self.profile_.embs[i], vec) for i in range(self.profile_.n)]
        temp = [self.profile_.scores[i, winner_j] * temp[i] for i in range(self.profile_.n)]
        return temp
