import numpy as np
from embedded_voting.epistemicGenerators.ratingsgeneratorepistemic import RatingsGeneratorEpistemic
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicGroupedMix(RatingsGeneratorEpistemic):
    """
    A generator of scores such that voters are
    separated into different groups and the noise of
    an voter on an alternative is equal to the noise
    of his group plus his own independent noise.
    The noise of different groups can be correlated due
    to the group features.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    groups_features : list or np.ndarray
        The features of each group of voters.
        Should be of the same length than :attr:`group_sizes`.
        Each row of this matrix correspond to the features of a group.
    group_noise : float
        The variance used to sample the noise of each group.
    independent_noise : float
        The variance used to sample the independent noise of each voter.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    groups_sizes : np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    group_noise : float
        The variance used to sample the variances of each group.
    independent_noise : float
        The variance used to sample the independent noise of each voter.

    Examples
    --------
    >>> np.random.seed(42)
    >>> features = [[1, 0], [0, 1], [1, 1]]
    >>> generator = RatingsGeneratorEpistemicGroupedMix([2, 2, 2], features)
    >>> generator()
    (array([13.74540119]), Ratings([[14.81094637],
             [14.81094637],
             [13.41737103],
             [13.41737103],
             [14.1141587 ],
             [14.1141587 ]]))
    """
    def __init__(self, groups_sizes, groups_features, group_noise=1, independent_noise=0,
                 minimum_score=10, maximum_score=20):
        groups_sizes = np.array(groups_sizes)
        n_voters = int(groups_sizes.sum())
        super().__init__(n_voters, minimum_score, maximum_score)
        self.groups_sizes = groups_sizes
        self.groups_features = np.array(groups_features)
        self.group_noise = group_noise
        self.independent_noise = independent_noise

    def __call__(self, n_candidates=1, *args):
        scores = np.zeros((self.n_voters, n_candidates))
        truth = np.zeros(n_candidates)
        n_groups, n_features = self.groups_features.shape
        for i in range(n_candidates):
            truth_i = self.generate_true_score()
            truth[i] = truth_i
            sigma = np.abs(np.random.randn(n_groups) * self.group_noise)
            cov = np.zeros((n_features, n_features))
            for k in range(n_features):
                cov[k, k] = sigma[k]
            scores_groups = np.random.multivariate_normal(np.ones(n_features) * truth_i, cov)
            s = 0
            scores_i = np.zeros(self.n_voters)
            for k in range(n_groups):
                n_voters_k = self.groups_sizes[k]
                cat_val = np.dot(scores_groups, self.groups_features[k]) / np.sum(self.groups_features[k])
                scores_i[s:s + n_voters_k] = cat_val + np.random.randn(n_voters_k) * self.independent_noise
                s += n_voters_k
            scores[:, i] = scores_i

        return truth, Ratings(scores)
