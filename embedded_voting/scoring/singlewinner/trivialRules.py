import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringFunction


class RangeVoting(ScoringFunction):
    """
    Voting rule that rank candidates by the sum of voters' scores

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election


    """
    def score_(self, cand):
        return self.profile_.scores[::, cand].sum()


class Nash(ScoringFunction):
    """
    Voting rule that rank candidates by the products of voters' score

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election


    """
    def score_(self, cand):
        scores = self.profile_.scores[::, cand]
        count = 0
        prod = 1
        for s in scores:
            if s > 0:
                count += 1
                prod *= s
        return (count, prod)


    @cached_property
    def ranking_(self):
        rank = [s[0] for s in self.scores_]
        scores = [s[1] for s in self.scores_]
        return np.lexsort((scores, rank))[::-1]
