import numpy as np
from embedded_voting.epistemicGenerators.ratings_generator_epistemic \
    import RatingsGeneratorEpistemic
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupedMean(RatingsGeneratorEpistemic):
    """
    A generator of ratings such that voters are
    separated into different groups and the noise of
    an voter on an alternative is equal to the noise
    of his group plus his own independent noise.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.RatingsGenerator.n_voters`.
    group_noise : float
        The variance used to sample the noise of each group.
    independent_noise : float
        The variance used to sample the independent noise of each voter.
    minimum_value : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_value : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.

    Examples
    --------
    >>> np.random.seed(44)
    >>> generator = RatingsGeneratorEpistemicGroupedMean([2, 2])
    >>> generator()
    Ratings([[17.62582802],
             [17.62582802],
             [17.42216872],
             [17.42216872]])
    >>> generator.ground_truth_
    array([18.34842149])
    """
    def __init__(self, groups_sizes, group_noise=1, independent_noise=0, minimum_value=10,
                 maximum_value=20):
        super().__init__(minimum_value=minimum_value, maximum_value=maximum_value,
                         groups_sizes=groups_sizes)
        self.group_noise = group_noise
        self.independent_noise = independent_noise

    def __call__(self, n_candidates=1, *args):
        self.ground_truth_ = self.generate_true_values(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            sigma = np.abs(np.random.normal(loc=0, scale=self.group_noise, size=self.n_groups))
            v_group_noise = self.m_voter_group @ np.random.multivariate_normal(
                mean=np.zeros(self.n_groups), cov=np.diag(sigma))
            v_independent_noise = np.random.normal(
                loc=0, scale=self.independent_noise, size=self.n_voters)
            ratings[:, i] = self.ground_truth_[i] + v_group_noise + v_independent_noise
        return Ratings(ratings)
