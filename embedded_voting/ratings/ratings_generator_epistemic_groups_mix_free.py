import numpy as np
from embedded_voting.ratings.ratings_generator_epistemic_groups import RatingsGeneratorEpistemicGroups
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupsMixFree(RatingsGeneratorEpistemicGroups):

    def __init__(self,
                 groups_sizes,
                 groups_features,
                 group_noise=1,
                 independent_noise=0,
                 truth_generator=None,
                 center_gap=100,
                 max_scale=1,
                 group_noise_f=None,
                 independent_noise_f=None):
        super().__init__(truth_generator=truth_generator, groups_sizes=groups_sizes)
        self.groups_features = np.array(groups_features)
        self.groups_features_normalized = (
            self.groups_features
            / self.groups_features.sum(1)[:, np.newaxis]
        )
        n_groups = len(groups_sizes)
        self.group_noise = group_noise
        self.independent_noise = independent_noise
        _, self.n_features = self.groups_features.shape

        if group_noise_f is None:
            group_noise_f = np.random.normal
        self.group_noise_f = group_noise_f
        if independent_noise_f is None:
            independent_noise_f = np.random.normal
        self.independent_noise_f = independent_noise_f

        self.centers = (np.random.rand(self.n_features)-0.5)*center_gap
        self.scales = 1 + np.random.rand(self.n_features)*(max_scale-1)

        noise_rescale_groups = self.groups_features.sum(1)/np.linalg.norm(self.groups_features, axis=1)
        self.noise_rescale = np.zeros(self.n_voters)
        curr_ind = 0
        for i in range(n_groups):
            self.noise_rescale[curr_ind:curr_ind + groups_sizes[i]] = noise_rescale_groups[i]
            curr_ind += groups_sizes[i]

    def __call__(self, n_candidates=1):
        self.ground_truth_ = self.truth_generator(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            noise_features = self.centers + self.group_noise_f(size=self.n_features)*self.group_noise*self.scales
            v_noise_dependent = (
                self.m_voter_group
                @ self.groups_features_normalized
                @ noise_features
            )
            v_noise_independent = self.independent_noise_f(size=self.n_voters)*self.independent_noise
            ratings[:, i] = self.ground_truth_[i] + v_noise_dependent*self.noise_rescale + v_noise_independent
        return Ratings(ratings)
