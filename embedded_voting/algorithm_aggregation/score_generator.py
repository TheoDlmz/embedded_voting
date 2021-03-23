
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class ScoreGenerator:
    """
    A generator of scores for a group of algorithms.

    Parameters
    ----------
    n_voters : int
        The number of voters in the generator.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    n_voters : int
        The number of voters in the generator.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    """

    def __init__(self, n_voters=1, minimum_score=10, maximum_score=20):
        self.n_voters = n_voters
        self.minimum_score = minimum_score
        self.maximum_score = maximum_score

    def generate_true_score(self):
        """
        This function generate a true value for some alternative.

        Return
        ------
        float
            The true value of an alternative
        """
        return self.minimum_score + np.random.rand()*(self.maximum_score-self.minimum_score)

    def sample_scores(self, n_alternatives=1):
        """
        This function computes the true values and the scores given by each voter to
        different alternative.

        Parameters
        ----------
        n_alternatives : int
            The number of alternatives of which we want the score.

        Return
        ------
        np.ndarray
            A list of length ``n_alternatives`` containing the true values of each alternative.
        np.ndarray
            A matrix of shape :attr:`~embedded_voting.ScoreGenerator.n_voters`, ``n_alternatives`` containing the scores
            given by each voter to each alternatives.
        """
        raise NotImplementedError

    def plot_scores(self, show=True):
        true_value, scores = self.sample_scores()
        plt.plot([true_value[0]]*2, [0, 1], color="red", label="True value")
        for i in range(self.n_voters):
            plt.plot([scores[i]]*2, [0, 1], color="black")
        plt.ylim(0, 1)
        plt.title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()


class MultivariateGenerator(ScoreGenerator):
    """
    A generator of scores based on a covariance matrix.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix of the voters.
        Should be of shape :attr:`~embedded_voting.ScoreGenerator.n_voters`,
        :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    independent_noise : float
        The variance of the independent noise.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix of the voters.
        Should be of shape :attr:`~embedded_voting.ScoreGenerator.n_voters`,
        :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    independent_noise : float
        The variance of the independent noise.

    """
    def __init__(self, covariance_matrix, independent_noise=0, minimum_score=10, maximum_score=20):
        self.covariance_matrix = covariance_matrix
        self.independent_noise = independent_noise
        n_voters = len(covariance_matrix)
        super().__init__(n_voters, minimum_score, maximum_score)

    def set_noise(self, independent_noise):
        """
        Update the :attr:`independent_noise` of the model.

        Parameters
        ----------
        independent_noise : float
            The new noise.

        Return
        ------
        MultivariateGenerator
            The object itself
        """
        self.independent_noise = independent_noise
        return self

    def sample_scores(self, n_alternatives=1):
        scores = np.zeros((self.n_voters, n_alternatives))
        truth = np.zeros(n_alternatives)
        for i in range(n_alternatives):
            truth_i = self.generate_true_score()
            truth[i] = truth_i
            scores_i = np.random.multivariate_normal(np.ones(self.n_voters)*truth_i, self.covariance_matrix)
            scores_i += np.random.randn(self.n_voters) * self.independent_noise
            scores[:, i] = scores_i

        return truth, scores


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

    """
    def __init__(self, groups_sizes, group_noise=0, minimum_score=10, maximum_score=20):
        self.groups_sizes = np.array(groups_sizes)
        self.group_noise = group_noise
        n_voters = int(self.groups_sizes.sum())
        super().__init__(n_voters, minimum_score, maximum_score)

    def set_noise(self, group_noise):
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

    def sample_scores(self, n_alternatives=1):
        scores = np.zeros((self.n_voters, n_alternatives))
        truth = np.zeros(n_alternatives)
        for i in range(n_alternatives):
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

    def plot_scores(self, show=True):
        true_value, scores = self.sample_scores()
        count = 0
        n_group = -1
        color = cm.rainbow(np.linspace(0, 0.8, len(self.groups_sizes)))
        plt.plot([true_value[0]]*2, [0, 1], color="red", label="True value")
        for i in range(self.n_voters):
            if i >= count:
                count += self.groups_sizes[n_group+1]
                n_group += 1
                plt.plot([scores[i]]*2, [0, 1], color=color[n_group], label="group %i" % (n_group+1))
            else:
                plt.plot([scores[i]]*2, [0, 1], color=color[n_group])
        plt.ylim(0, 1)
        plt.title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()


class GroupedMeanGenerator(ScoreGenerator):
    """
    A generator of scores such that voters are
    separated into different groups and the noise of
    an voter on an alternative is equal to the noise
    of his group plus his own independent noise.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.ScoreGenerator.n_voters`.
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

    """
    def __init__(self, groups_sizes, group_noise=0, independent_noise=0, minimum_score=10, maximum_score=20):
        self.groups_sizes = np.array(groups_sizes)
        self.group_noise = group_noise
        self.independent_noise = independent_noise
        n_voters = int(self.groups_sizes.sum())
        super().__init__(n_voters, minimum_score, maximum_score)

    def set_group_noise(self, group_noise):
        """
        Update the :attr:`group_noise` of the model.

        Parameters
        ----------
        group_noise : float
            The new noise.

        Return
        ------
        GroupedMeanGenerator
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
        GroupedMeanGenerator
            The object itself
        """
        self.independent_noise = independent_noise
        return self

    def sample_scores(self, n_alternatives=1):
        scores = np.zeros((self.n_voters, n_alternatives))
        truth = np.zeros(n_alternatives)
        for i in range(n_alternatives):
            truth_i = self.generate_true_score()
            truth[i] = truth_i
            sigma = np.abs(np.random.randn(len(self.groups_sizes)) * self.group_noise)
            cov = np.zeros((self.n_voters, self.n_voters))
            s = 0
            for k in range(len(self.groups_sizes)):
                cov[s:s + self.groups_sizes[k], s:s + self.groups_sizes[k]] = np.ones(self.groups_sizes[k]) * sigma[k]
                s += self.groups_sizes[k]
            scores_i = np.random.multivariate_normal(np.ones(self.n_voters) * truth_i, cov)
            scores_i += np.random.randn(self.n_voters) * self.independent_noise
            scores[:, i] = scores_i

        return truth, scores

    def plot_scores(self, show=True):
        true_value, scores = self.sample_scores()
        count = 0
        n_group = -1
        color = cm.rainbow(np.linspace(0, 0.8, len(self.groups_sizes)))
        plt.plot([true_value[0]]*2, [0, 1], color="red", label="True value")
        for i in range(self.n_voters):
            if i >= count:
                count += self.groups_sizes[n_group+1]
                n_group += 1
                plt.plot([scores[i]]*2, [0, 1], color=color[n_group], label="group %i" % (n_group+1))
            else:
                plt.plot([scores[i]]*2, [0, 1], color=color[n_group])
        plt.ylim(0, 1)
        plt.title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()


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
    groups_features : np.ndarray
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

    """
    def __init__(self, groups_sizes, groups_features, group_noise=0, independent_noise=0,
                 minimum_score=10, maximum_score=20):
        self.groups_sizes = np.array(groups_sizes)
        self.groups_features = np.array(groups_features)
        self.group_noise = group_noise
        self.independent_noise = independent_noise
        n_voters = int(self.groups_sizes.sum())
        super().__init__(n_voters, minimum_score, maximum_score)

    def set_group_noise(self, group_noise):
        """
        Update the :attr:`group_noise` of the model.

        Parameters
        ----------
        group_noise : float
            The new noise.

        Return
        ------
        GroupedMeanGenerator
            The object itself
        """
        self.group_noise = group_noise

    def set_independent_noise(self, independent_noise):
        """
        Update the :attr:`independent_noise` of the model.

        Parameters
        ----------
        independent_noise : float
            The new noise.

        Return
        ------
        GroupedMeanGenerator
            The object itself
        """
        self.independent_noise = independent_noise

    def sample_scores(self, n_alternatives=1):
        scores = np.zeros((self.n_voters, n_alternatives))
        truth = np.zeros(n_alternatives)
        n_groups, n_features = self.groups_features.shape
        for i in range(n_alternatives):
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

    def plot_scores(self, show=True):
        true_value, scores = self.sample_scores()
        count = 0
        n_group = -1
        color = cm.rainbow(np.linspace(0, 0.8, len(self.groups_sizes)))
        plt.plot([true_value[0]]*2, [0, 1], color="red", label="True value")
        for i in range(self.n_voters):
            if i >= count:
                count += self.groups_sizes[n_group+1]
                n_group += 1
                plt.plot([scores[i]]*2, [0, 1], color=color[n_group], label="group %i" % (n_group+1))
            else:
                plt.plot([scores[i]]*2, [0, 1], color=color[n_group])
        plt.ylim(0, 1)
        plt.title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()
