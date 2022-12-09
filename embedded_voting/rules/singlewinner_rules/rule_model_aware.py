import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property


class RuleModelAware(Rule):
    """
    A rule that is aware of settings of the ratings generator (i.e. the algorithms) and use the maximum likelihood
    to select the best candidate.
    """

    def __init__(self,
                 groups_sizes,
                 groups_features,
                 group_noise=1,
                 independent_noise=0):

        super().__init__(score_components=1)
        self.groups_sizes = np.array(groups_sizes, dtype=int)
        self.n_voters = self.groups_sizes.sum()
        self.n_groups = self.groups_sizes.size

        self.groups_features = np.array(groups_features)
        self.groups_features_normalized = (
            self.groups_features
            / self.groups_features.sum(1)[:, np.newaxis]
        )
        noise_rescale_groups = self.groups_features.sum(1)/np.linalg.norm(self.groups_features, axis=1)

        self.m_voter_group = np.vstack([
            np.hstack((
                np.zeros((group_size, i_group)),
                np.ones((group_size, 1))*noise_rescale_groups[i_group],
                np.zeros((group_size, self.n_groups - i_group - 1))
            ))
            for i_group, group_size in enumerate(self.groups_sizes)
        ])

        v_noise_dependent = (
            self.m_voter_group
            @ self.groups_features_normalized
        )*group_noise
        v_noise_independent = np.eye(self.n_voters)*independent_noise

        v_noise = np.concatenate([v_noise_dependent, v_noise_independent], axis=1)
        if independent_noise == 0:
            self.sigma_inv = (self.m_voter_group/self.m_voter_group.sum(0)).T
        else:
            self.sigma_inv = np.linalg.inv(np.dot(v_noise, v_noise.T))

    @cached_property
    def weights_(self):
        return self.sigma_inv.sum(axis=0)

    def _score_(self, candidate):
        return self.ratings_.candidate_ratings(candidate) @ self.weights_
