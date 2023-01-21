import numpy as np
from embedded_voting.ratings.ratings_generator_epistemic_groups import RatingsGeneratorEpistemicGroups
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupsMix(RatingsGeneratorEpistemicGroups):
    """
    A generator of ratings such that voters are
    separated into different groups and the noise of
    an voter on a candidate is equal to the noise
    of his group plus his own independent noise.
    The noise of different groups can be correlated due
    to the group features.

    For each candidate `i`:

    * For each feature, a `sigma_feature` is drawn (absolute part of a normal variable, scaled by
      `group_noise`). Then a `noise_feature` is drawn (normal variable scaled by `sigma_feature`).
    * For each group, `noise_group` is the barycenter of the values of `noise_feature`, with the
      weights for each feature given by `groups_features`.
    * For each voter, `noise_dependent` is equal to the `noise_group` of her group.
    * For each voter, `noise_independent` is drawn (normal variable scaled by `independent_noise`).
    * For each voter of each group, the rating is computed as
      `ground_truth[i] + noise_dependent + noise_independent`.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.RatingsGenerator.n_voters`.
    groups_features : list or np.ndarray
        The features of each group of voters.
        Should be of the same length than :attr:`group_sizes`.
        Each row of this matrix correspond to the features of a group.
    group_noise : float
        The variance used to sample the noise of each group.
    independent_noise : float
        The variance used to sample the independent noise of each voter.
    truth_generator : TruthGenerator
        The truth generator used to generate to true values of each candidate.
        Default: `TruthGeneratorUniform(10, 20)`.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.

    Examples
    --------
    >>> np.random.seed(42)
    >>> features = [[1, 0], [0, 1], [1, 1]]
    >>> generator = RatingsGeneratorEpistemicGroupsMix([2, 2, 2], features)
    >>> generator()  # doctest: +ELLIPSIS
    Ratings([[18.1960...],
             [18.1960...],
             [18.3058...],
             [18.3058...],
             [18.2509...],
             [18.2509...]])
    >>> generator.ground_truth_  # doctest: +ELLIPSIS
    array([17.7395...])

    >>> np.random.seed(42)
    >>> features = [[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]]
    >>> generator = RatingsGeneratorEpistemicGroupsMix([2, 2, 2], features)
    >>> generator() # doctest: +ELLIPSIS
    Ratings([[17.951...],
             [17.951...],
             [17.737...],
             [17.737...],
             [18.438...],
             [18.438...]])
    """
    def __init__(self, groups_sizes, groups_features, group_noise=1, independent_noise=0, truth_generator=None):
        super().__init__(truth_generator=truth_generator, groups_sizes=groups_sizes)
        self.groups_features = np.array(groups_features)
        self.groups_features_normalized = (
            self.groups_features
            / self.groups_features.sum(1)[:, np.newaxis]
        )
        self.group_noise = group_noise
        self.independent_noise = independent_noise
        _, self.n_features = self.groups_features.shape

    def __call__(self, n_candidates=1):
        self.ground_truth_ = self.truth_generator(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            sigma_features = np.abs(
                np.random.normal(loc=0, scale=self.group_noise, size=self.n_features)
            )
            noise_features = np.random.multivariate_normal(
                mean=np.zeros(self.n_features), cov=np.diag(sigma_features))
            v_noise_dependent = (
                self.m_voter_group
                @ self.groups_features_normalized
                @ noise_features
            )
            v_noise_independent = np.random.normal(
                loc=0, scale=self.independent_noise, size=self.n_voters)
            ratings[:, i] = self.ground_truth_[i] + v_noise_dependent + v_noise_independent
        return Ratings(ratings)
