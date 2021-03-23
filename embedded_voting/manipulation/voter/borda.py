import numpy as np
from embedded_voting.manipulation.voter.general import SingleVoterManipulationExtension
from embedded_voting.scoring.singlewinner.ordinal import BordaExtension
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.svd import SVDNash


class SingleVoterManipulationBorda(SingleVoterManipulationExtension):
    """
    This class do the single voter manipulation
    analysis for the :class:`BordaExtension` extension.
    It is faster than the general class
    class:`SingleVoterManipulationExtension`.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> manipulation = SingleVoterManipulationBorda(my_profile, SVDNash())
    >>> manipulation.prop_manipulator_
    0.1
    >>> manipulation.avg_welfare_
    0.9
    >>> manipulation.worst_welfare_
    0.0
    >>> manipulation.manipulation_global_
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    def __init__(self, profile, rule=None):
        super().__init__(profile, BordaExtension(profile), rule)

    def manipulation_voter(self, i):
        fake_scores_i = self.extended_rule.fake_profile.scores[i].copy()
        score_i = self.profile_.scores[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        m = self.profile_.n_candidates

        if preferences_order[0] == self.winner_:
            return self.winner_

        all_scores = []
        for e in range(m):
            self.extended_rule.fake_profile.scores[i] = np.ones(self.profile_.n_candidates) * (e / (m - 1))
            altered_scores = self.extended_rule.base_rule(self.extended_rule.fake_profile).scores_
            all_scores += [(s, j, e) for j, s in enumerate(altered_scores)]

        self.extended_rule.fake_profile.scores[i] = fake_scores_i
        all_scores.sort()
        all_scores = all_scores[::-1]

        buckets = np.arange(m)

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]
        for (_, j, kind) in all_scores:
            buckets[kind] -= 1
            if buckets[kind] < 0:
                break

            if kind == (m-1):
                index_candidate = np.where(preferences_order == j)[0][0]
                if index_candidate < best_manipulation_i:
                    best_manipulation_i = index_candidate

        best_manipulation = preferences_order[best_manipulation_i]

        return best_manipulation
