
import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property


class Manipulation(DeleteCacheMixin):
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

    def manipulation_voter(self, i, verbose=False):
        score_i = self.profile_.scores[i].copy()
        pref_ord = np.argsort(score_i)[::-1]

        if pref_ord[0] == self.winner_:
            return self.winner_

        self.profile_.scores[i] = np.ones(self.profile_.m)
        scores_max = self.rule_(self.profile_).scores_
        self.profile_.scores[i] = np.zeros(self.profile_.m)
        scores_min = self.rule_(self.profile_).scores_
        self.profile_.scores[i] = score_i

        all_scores = [(s, i, 1) for i, s in enumerate(scores_max)] \
                     + [(s, i, 0) for i, s in enumerate(scores_min)]

        all_scores.sort()
        all_scores = all_scores[::-1]

        if verbose:
            print(i, self.winner_, all_scores)

        best_manip = np.where(pref_ord == self.winner_)[0][0]
        for (_, i, k) in all_scores:
            if k == 0:
                break

            index_cand = np.where(pref_ord == i)[0][0]
            if index_cand < best_manip:
                best_manip = index_cand

        best_manip = pref_ord[best_manip]

        if verbose:
            print(self.welfare_[best_manip])

        return best_manip

    @cached_property
    def manipulation_global_(self):
        return [self.manipulation_voter(i) for i in range(self.profile_.n)]

    @cached_property
    def prop_manipulator_(self):
        return len([x for x in self.manipulation_global_ if x != self.winner_])/self.profile_.n

    @cached_property
    def avg_welfare_(self):
        return np.mean([self.welfare_[x]for x in self.manipulation_global_])

    @cached_property
    def worst_welfare_(self):
        return np.min([self.welfare_[x]for x in self.manipulation_global_])

    @cached_property
    def is_manipulable_(self):
        for i in range(self.profile_.n):
            if self.manipulation_voter(i) != self.winner_:
                return True
        return False

class ManipulationExtension(Manipulation):
    def __init__(self, profile=None, extension=None):
        self.profile_ = profile
        self.extension = extension

    def __call__(self, r):
        self.rule_ = r
        self.extended_rule = self.extension.set_rule(r)
        self.extended_rule(self.profile_)
        self.winner_ = self.extended_rule.winner_
        self.welfare_ = self.rule_(self.profile_).welfare_
        self.delete_cache()
        return self

    def manipulation_voter(self, i, verbose=False):
        score_i = self.profile_.scores[i].copy()
        pref_ord = np.argsort(score_i)[::-1]
        points = np.arange(self.profile_.m)[::-1]
        if pref_ord[0] == self.winner_:
            return self.winner_

        best_manip_i = np.where(pref_ord == self.winner_)[0][0]

        for perm in itertools.permutations(range(self.profile_.m)):
            self.profile_.scores[i] = points[list(perm)]
            fake_run = self.extended_rule(self.profile_)
            new_winner = fake_run.winner_
            index_cand = np.where(pref_ord == new_winner)[0][0]
            if index_cand < best_manip_i:
                best_manip_i = index_cand
                if best_manip_i == 0:
                    break

        best_manip = pref_ord[best_manip_i]
        self.profile_.scores[i] = score_i

        if verbose:
            print(i, self.welfare_[best_manip])

        return best_manip
