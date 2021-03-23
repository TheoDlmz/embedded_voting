import numpy as np
from embedded_voting.manipulation.voter.general import SingleVoterManipulationExtension
from embedded_voting.scoring.singlewinner.ordinal import InstantRunoffExtension
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.svd import SVDNash


class SingleVoterManipulationIRV(SingleVoterManipulationExtension):
    """
    This class do the single voter manipulation
    analysis for the :class:`InstantRunoffExtension` extension.
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
    >>> manipulation = SingleVoterManipulationIRV(my_profile, SVDNash())
    >>> manipulation.prop_manipulator_
    0.0
    >>> manipulation.avg_welfare_
    1.0
    >>> manipulation.worst_welfare_
    1.0
    >>> manipulation.manipulation_global_
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """

    def __init__(self, profile, rule=None):
        super().__init__(profile, InstantRunoffExtension(profile), rule)

    def _create_fake_scores(self, eliminated, scores):
        """
        This function creates a fake profile for each step of the IRV function

        Parameters
        ----------
        eliminated : int list
            The list of candidates already eliminated.

        scores : float list
            The scores of the candidates.

        Return
        ------
        np.ndarray
            A fake score matrix of size :attr:`n_voter`, :attr:`n_candidate`.

        """
        fake_profile = np.zeros((self.profile_.n_voters, self.profile_.n_candidates))
        points = np.zeros(self.profile_.n_candidates)
        points[0] = 1

        for i in range(self.profile_.n_voters):
            scores_i = scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        return fake_profile

    def manipulation_voter(self, i):
        scores = self.profile_.scores.copy()
        score_i = self.profile_.scores[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        m = self.profile_.n_candidates

        if preferences_order[0] == self.winner_:
            return self.winner_

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]

        queue_eliminated = [([], -1)]

        while len(queue_eliminated) > 0:
            (el, one) = queue_eliminated.pop()

            if len(el) == m:
                winner = el[-1]
                index_candidate = np.where(preferences_order == winner)[0][0]
                if index_candidate < best_manipulation_i:
                    best_manipulation_i = index_candidate
            else:
                fake_profile = self._create_fake_scores(el, scores)
                fake_profile[i] = np.ones(self.profile_.n_candidates)
                self.profile_.scores = fake_profile
                scores_max = self.extended_rule.rule_(self.profile_).scores_

                fake_profile[i] = np.zeros(self.profile_.n_candidates)
                self.profile_.scores = fake_profile
                scores_min = self.extended_rule.rule_(self.profile_).scores_

                all_scores = [(s, j, 1) for j, s in enumerate(scores_max) if j not in el]
                all_scores += [(s, j, 0) for j, s in enumerate(scores_min) if j not in el]

                all_scores.sort()

                if all_scores[0][1] == one:
                    if all_scores[1][1] == one:
                        queue_eliminated.append((el+[all_scores[0][1]], -1))
                    else:
                        queue_eliminated.append((el+[all_scores[1][1]], one))
                else:
                    queue_eliminated.append((el+[all_scores[0][1]], one))
                    if all_scores[1][2] == 0 and one == -1:
                        queue_eliminated.append((el+[all_scores[1][1]], all_scores[0][1]))

        self.profile_.scores = scores

        best_manipulation = preferences_order[best_manipulation_i]

        return best_manipulation
