
import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property


class ManipulationCoalition(DeleteCacheMixin):
    def __init__(self, profile=None):
        self.profile_ = profile

    def __call__(self, r):
        self.rule_ = r
        global_rule = self.rule_(self.profile_)
        self.winner_ = global_rule.winner_
        self.scores_ = global_rule.scores_
        self.welfare_ = global_rule.welfare_
        self.delete_cache()
        return self

    def manipulation_triviale(self, cand, verbose=False):

        voters_interested = []
        for i in range(self.profile_.n):
            score_i = self.profile_.scores[i]
            if score_i[self.winner_] < score_i[cand]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i"%(len(voters_interested), cand, self.winner_))

        old_profile = self.profile_.scores.copy()
        for i in voters_interested:
            self.profile_.scores[i] = np.zeros(self.profile_.m)
            self.profile_.scores[i][cand] = 1

        new_winner = self.rule_(self.profile_).winner_
        self.profile_.scores = old_profile

        if verbose:
            print("Winner is %i"%new_winner)
        return (new_winner == cand)

    @cached_property
    def is_manipulable_(self):
        for i in range(self.profile_.m):
            if i == self.winner_:
                continue
            if self.manipulation_triviale(i):
                return True
        return False

    @cached_property
    def worst_welfare_(self):
        welf = self.welfare_[self.winner_]
        for i in range(self.profile_.m):
            if i == self.winner_:
                continue
            if self.manipulation_triviale(i):
                welf = min(welf, self.welfare_[i])
        return welf
