import numpy as np
from embedded_voting.epistemicGenerators.ratingsgeneratorepistemic import RatingsGeneratorEpistemic
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupedNoise(RatingsGeneratorEpistemic):
    """
    A generator of scores such that voters are separated into different groups and for each
    alternative the variance of each voter of the same group is the same.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    group_noise : float
        The variance used to sample the variances of each group.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth scores of the candidates corresponding to the
        last Ratings generated

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = RatingsGeneratorEpistemicGroupedNoise([2, 2])
    >>> generator()
    Ratings([[14.03963831],
             [14.81094637],
             [13.41737103],
             [13.44883031]])
    >>> generator.ground_truth_
    array([13.74540119])
    """
    def __init__(self, groups_sizes, group_noise=1, minimum_score=10, maximum_score=20):
        super().__init__(minimum_score=minimum_score, maximum_score=maximum_score,
                         groups_sizes=groups_sizes)
        self.group_noise = group_noise

    def __call__(self, n_candidates=1, *args):
        self.ground_truth_ = self.generate_true_values(n_candidates=n_candidates)
        scores = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            sigma = np.abs(np.random.randn(len(self.groups_sizes)) * self.group_noise)
            cov = np.zeros((self.n_voters, self.n_voters))
            s = 0
            for k in range(len(self.groups_sizes)):
                cov[s:s + self.groups_sizes[k], s:s + self.groups_sizes[k]] = np.eye(self.groups_sizes[k]) * sigma[k]
                s += self.groups_sizes[k]
            scores_i = np.random.multivariate_normal(np.ones(self.n_voters)*self.ground_truth_[i], cov)
            scores[:, i] = scores_i
        return Ratings(scores)
