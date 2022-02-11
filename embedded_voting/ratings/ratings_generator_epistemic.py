
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from embedded_voting.ratings.ratings_generator import RatingsGenerator
from embedded_voting.truth.truth_generator import TruthGenerator
from embedded_voting.truth.truth_generator_uniform import TruthGeneratorUniform


class RatingsGeneratorEpistemic(RatingsGenerator):
    """
    A generator of ratings based on a ground truth ("true value") for each candidate.

    Parameters
    ----------
    n_voters : int
        The number of voters in the generator.
    truth_generator : TruthGenerator
        The truth generator used to generate to true values of each candidate.
        Default: `TruthGeneratorUniform(10, 20)`.
    groups_sizes : list or np.ndarray
        The number of voters in each group.
        If set to None, then there are no "groups".

    Attributes
    ----------
    n_groups : int
        The number of groups. If `groups_size` is None, then `n_groups` is None.
    m_voter_group : np.ndarray
        Incidence matrix between voters and groups: `m_voter_group[v, g]` is 1 if
        and only if voter `v` is in group `g`, and 0 otherwise.
        Size `n_voters` * `n_groups`.
        If `groups_size` is None, then `m_voter_group` is None.
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.
    """

    def __init__(self, n_voters=None, truth_generator=None, groups_sizes=None):
        if truth_generator is None:
            truth_generator = TruthGeneratorUniform(minimum_value=10, maximum_value=20)
        self.truth_generator = truth_generator
        if groups_sizes is not None:
            groups_sizes = np.array(groups_sizes)
            n_voters_computed = np.sum(groups_sizes)
            if n_voters is not None and n_voters != n_voters_computed:
                raise ValueError('n_voters should be equal to the sum of groups_sizes.')
            n_voters = n_voters_computed
            self.n_groups = len(groups_sizes)
            self.m_voter_group = np.vstack([
                np.hstack((
                    np.zeros((group_size, i_group)),
                    np.ones((group_size, 1)),
                    np.zeros((group_size, self.n_groups - i_group - 1))
                ))
                for i_group, group_size in enumerate(groups_sizes)
            ])
        else:
            self.n_groups = None
            self.m_voter_group = None
        super().__init__(n_voters)
        self.groups_sizes = groups_sizes
        self.ground_truth_ = None

    def generate_true_values(self, n_candidates=1):
        """
        This function generate a true value for each candidate.

        Return
        ------
        np.ndarray
            The true value for each candidate.
        """
        return self.truth_generator(n_candidates)

    def __call__(self, n_candidates=1, *args):
        """
        This function generate the ground truth and the ratings given by each voter to
        n_candidates candidates.

        Parameters
        ----------
        n_candidates : int
            The number of candidates of which we want the ratings.

        Return
        ------
        Ratings
            The ratings given by the voters to the alternatives
        """
        raise NotImplementedError

    def plot_ratings(self, show=True):
        """
        This function plots the true value of a candidate and the ratings
        given by each voter for some candidate randomly selected.

        Parameters
        ----------
        show : bool
            If True, displays the plot at the end of the function.
        """
        ratings = self()
        fig, ax = plt.subplots()
        ax.plot([self.ground_truth_[0]] * 2, [0, 1], color="red", label="True value")
        if self.groups_sizes is None:
            for i_voter in range(self.n_voters):
                ax.plot([ratings[i_voter]] * 2, [0, 1], color="k")
        else:
            # noinspection PyUnresolvedReferences
            color = cm.rainbow(np.linspace(0, 0.8, self.n_groups))
            sum_previous_groups_sizes = 0
            for i_group, group_size in enumerate(self.groups_sizes):
                for i_voter_in_group in range(group_size):
                    i_voter = sum_previous_groups_sizes + i_voter_in_group
                    if i_voter_in_group == 0:
                        ax.plot([ratings[i_voter]] * 2, [0, 1], color=color[i_group],
                                label="group %i" % (i_group + 1))
                    else:
                        ax.plot([ratings[i_voter]] * 2, [0, 1], color=color[i_group])
                sum_previous_groups_sizes += group_size
        ax.set_ylim(0, 1)
        ax.set_title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()  # pragma: no cover
