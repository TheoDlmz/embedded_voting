import numpy as np
from embedded_voting.ratings.ratings_generator_epistemic_groups \
    import RatingsGeneratorEpistemicGroups
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupsNoise(RatingsGeneratorEpistemicGroups):
    """
    A generator of ratings such that voters are separated into different groups and for each
    candidate the variance of each voter of the same group is the same.

    For each candidate `i`:

    * For each group, a `sigma_group` is drawn (absolute part of a normal variable, scaled by
      `group_noise`).
    * For each voter, her `sigma_voter` is equal to the `sigma_group` of her group. Her
      `noise_voter` is drawn (normal variable scaled by `sigma_voter`).
    * For each voter, the rating is computed as `ground_truth[i] + noise_voter`.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.RatingsGenerator.n_voters`.
    group_noise : float
        The variance used to sample the variances of each group.
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
    >>> generator = RatingsGeneratorEpistemicGroupsNoise([2, 2])
    >>> generator()  # doctest: +ELLIPSIS
    Ratings([[18.196...],
             [18.812...],
             [17.652...],
             [17.652...]])
    >>> generator.ground_truth_  # doctest: +ELLIPSIS
    array([17.739...])
    """
    def __init__(self, groups_sizes, group_noise=1, truth_generator=None):
        super().__init__(truth_generator=truth_generator, groups_sizes=groups_sizes)
        self.group_noise = group_noise

    def __call__(self, n_candidates=1):
        self.ground_truth_ = self.truth_generator(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            sigma_groups = np.abs(np.random.normal(loc=0, scale=self.group_noise, size=self.n_groups))
            sigma_voters = self.m_voter_group @ sigma_groups
            v_noise = np.random.multivariate_normal(
                mean=np.zeros(self.n_voters), cov=np.diag(sigma_voters))
            ratings[:, i] = self.ground_truth_[i] + v_noise
        return Ratings(ratings)
