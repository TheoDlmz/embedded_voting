import numpy as np
from matplotlib.pyplot import cm
from embedded_voting.ratings.ratings_generator_epistemic import RatingsGeneratorEpistemic


class RatingsGeneratorEpistemicGroups(RatingsGeneratorEpistemic):
    """
    An epistemic generator based on a notion of groups of voters.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each group.
    truth_generator : TruthGenerator
        The truth generator used to generate to true values of each candidate.
        Default: `TruthGeneratorUniform(10, 20)`.

    Attributes
    ----------
    n_groups : int
        Number of groups.
    m_voter_group : np.ndarray
        Incidence matrix between voters and groups: `m_voter_group[v, g]` is 1 if
        and only if voter `v` is in group `g`, and 0 otherwise.
        Size `n_voters` * `n_groups`.
    """

    def __init__(self, groups_sizes, truth_generator=None):
        self.groups_sizes = np.array(groups_sizes)
        self.n_groups = self.groups_sizes.size
        self.m_voter_group = np.vstack([
            np.hstack((
                np.zeros((group_size, i_group)),
                np.ones((group_size, 1)),
                np.zeros((group_size, self.n_groups - i_group - 1))
            ))
            for i_group, group_size in enumerate(self.groups_sizes)
        ])
        super().__init__(n_voters=self.m_voter_group.shape[0], truth_generator=truth_generator)

    def __call__(self, n_candidates=1):
        raise NotImplementedError

    def _plot_ratings_aux(self, ax, ratings):
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
