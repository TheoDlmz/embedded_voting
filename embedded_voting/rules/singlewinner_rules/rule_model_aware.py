import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property
from embedded_voting.ratings.ratings import Ratings


class RuleModelAware(Rule):
    """
    A rule that is know the noise parameters of the model and use the maximum likelihood
    to select the best candidate.

    Parameters
    ----------
    groups_sizes : list of int
        The number of voters in each group.
    groups_features : np.ndarray of shape (n_groups, n_features)
        The features of each group.
    group_noise : float
        The value of the feature noise.
    independent_noise : float
        The value of the distinct noise.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> election = RuleModelAware([2, 1], [[1, 0], [0, 1]], group_noise=1, independent_noise=1)(ratings)
    >>> election.ranking_
    [1, 2, 0]
    >>> election.scores_
    [0.5, 0.7, 0.5666666...]
    >>> election.winner_
    1
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
