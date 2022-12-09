import numpy as np
from embedded_voting.ratings.ratings_generator_epistemic_groups import RatingsGeneratorEpistemicGroups
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupsMixAcc(RatingsGeneratorEpistemicGroups):

    def __init__(self,
                 groups_sizes,
                 groups_features,
                 group_noises,
                 independent_noise=0,
                 truth_generator=None,
                 center_gap=100,
                 max_scale=1):
        super().__init__(truth_generator=truth_generator, groups_sizes=groups_sizes)
        self.groups_features = np.array(groups_features)
        self.groups_features_normalized = (
            self.groups_features
            / self.groups_features.sum(1)[:, np.newaxis]
        )
        n_groups = len(groups_sizes)
        self.group_noises = group_noises
        self.independent_noise = independent_noise
        _, self.n_features = self.groups_features.shape
        self.centers = (np.random.rand(self.n_features) - 0.5) * center_gap
        self.scales = 1 + np.random.rand(self.n_features) * (max_scale - 1)

    def __call__(self, n_candidates=1):
        self.ground_truth_ = self.truth_generator(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            sigma_features = np.abs(
                np.random.normal(loc=0, size=self.n_features)*self.group_noises
            )
            noise_features = np.random.multivariate_normal(
                mean=self.centers, cov=np.diag(sigma_features * self.scales))
            v_noise_dependent = (
                self.m_voter_group
                @ self.groups_features_normalized
                @ noise_features
            )
            v_noise_independent = np.random.normal(
                loc=0, scale=self.independent_noise, size=self.n_voters)
            ratings[:, i] = self.ground_truth_[i] + v_noise_dependent + v_noise_independent
        return Ratings(ratings)
