import numpy as np
from embedded_voting.generators.scoregenerator import ScoreGenerator


class GroupedMixGenerator(ScoreGenerator):
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
    >>> generator = GroupedMixGenerator([2, 2, 2], features)
    >>> generator.sample_scores()
    (array([13.74540...]), array([[14.81094...],
           [14.81094...],
           [13.41737...],
           [13.41737...],
           [14.11415...],
           [14.11415...]]))
    >>> generator.set_group_noise(5).sample_scores()
    (array([11.39493...]), array([[15.44060...],
           [15.44060...],
           [10.91386...],
           [10.91386...],
           [13.17723...],
           [13.17723...]]))
    >>> generator.set_independent_noise(0.5).sample_scores()
    (array([10.65051...]), array([[8.79988...],
           [8.61393...],
           [9.26444...],
           [9.15248...],
           [9.26866...],
           [8.44142...]]))

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

    def set_group_noise(self, group_noise):
        """
        Update the :attr:`group_noise` of the model.

        Parameters
        ----------
        group_noise : float
            The new noise.

        Return
        ------
        GroupedMixGenerator
            The object itself
        """
        self.group_noise = group_noise
        return self

    def set_independent_noise(self, independent_noise):
        """
        Update the :attr:`independent_noise` of the model.

        Parameters
        ----------
        independent_noise : float
            The new noise.

        Return
        ------
        GroupedMixGenerator
            The object itself
        """
        self.independent_noise = independent_noise
        return self

    def sample_scores(self, n_candidates=1):
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

        return truth, scores
