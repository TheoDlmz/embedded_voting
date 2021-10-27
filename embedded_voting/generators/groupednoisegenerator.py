import numpy as np
from embedded_voting.generators.scoregenerator import ScoreGenerator


class GroupedNoiseGenerator(ScoreGenerator):
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
    groups_sizes : np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    group_noise : float
        The variance used to sample the variances of each group.

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = GroupedNoiseGenerator([2, 2])
    >>> generator.sample_scores()
    (array([13.74540...]), array([[14.03963...],
           [14.81094...],
           [13.41737...],
           [13.44883...]]))
    >>> generator.set_group_noise(5).sample_scores()
    (array([11.8340...]), array([[10.72001...],
           [12.17969...],
           [ 6.49894...],
           [10.09491...]]))
    """
    def __init__(self, groups_sizes, group_noise=1, minimum_score=10, maximum_score=20):
        groups_sizes = np.array(groups_sizes)
        n_voters = int(groups_sizes.sum())
        super().__init__(n_voters, minimum_score, maximum_score)
        self.groups_sizes = groups_sizes
        self.group_noise = group_noise

    def set_group_noise(self, group_noise):
        """
        Update the :attr:`group_noise` of the model.

        Parameters
        ----------
        group_noise : float
            The new noise.

        Return
        ------
        GroupedNoiseGenerator
            The object itself
        """
        self.group_noise = group_noise
        return self

    def sample_scores(self, n_candidates=1):
        scores = np.zeros((self.n_voters, n_candidates))
        truth = np.zeros(n_candidates)
        for i in range(n_candidates):
            truth_i = self.generate_true_score()
            truth[i] = truth_i
            sigma = np.abs(np.random.randn(len(self.groups_sizes)) * self.group_noise)
            cov = np.zeros((self.n_voters, self.n_voters))
            s = 0
            for k in range(len(self.groups_sizes)):
                cov[s:s + self.groups_sizes[k], s:s + self.groups_sizes[k]] = np.eye(self.groups_sizes[k]) * sigma[k]
                s += self.groups_sizes[k]
            scores_i = np.random.multivariate_normal(np.ones(self.n_voters)*truth_i, cov)
            scores[:, i] = scores_i
        return truth, scores
