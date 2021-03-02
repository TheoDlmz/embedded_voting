import numpy as np
from embedded_voting.manipulation.voter.general import ManipulationExtension


class ManipulationBorda(ManipulationExtension):

    def manipulation_voter(self, i, verbose=False):
        fake_scores_i = self.extended_rule.fake_profile_.scores[i].copy()
        score_i = self.profile_.scores[i].copy()
        pref_ord = np.argsort(score_i)[::-1]

        m = self.profile_.m

        if pref_ord[0] == self.winner_:
            return self.winner_

        all_scores = []
        for e in range(m):
            self.extended_rule.fake_profile_.scores[i] = np.ones(self.profile_.m)*(e/(m-1))
            scores_modif = self.extended_rule.base_rule(self.extended_rule.fake_profile_).scores_
            all_scores += [(s, j, e) for j, s in enumerate(scores_modif)]

        self.extended_rule.fake_profile_.scores[i] = fake_scores_i
        all_scores.sort()
        all_scores = all_scores[::-1]

        if verbose:
            print(i, self.winner_, all_scores)


        buckets = np.arange(m)

        best_manip_i = np.where(pref_ord == self.winner_)[0][0]
        for (_, j, kind) in all_scores:
            buckets[kind] -= 1
            if buckets[kind] < 0:
                break

            if kind == (m-1):
                index_cand = np.where(pref_ord == j)[0][0]
                if index_cand < best_manip_i:
                    best_manip_i = index_cand

        best_manip = pref_ord[best_manip_i]

        if verbose:
            print(self.welfare_[best_manip])

        return best_manip
