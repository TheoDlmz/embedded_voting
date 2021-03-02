import numpy as np
from embedded_voting.manipulation.voter.general import ManipulationExtension

class ManipulationApp(ManipulationExtension):

    def manipulation_voter(self, i, verbose=False):
        fake_scores_i = self.extended_rule.fake_profile_.scores[i].copy()
        score_i = self.profile_.scores[i].copy()
        pref_ord = np.argsort(score_i)[::-1]

        k = int(np.sum(self.extension.points))
        unk = self.profile_.m - k

        if pref_ord[0] == self.winner_:
            return self.winner_

        self.extended_rule.fake_profile_.scores[i] = np.ones(self.profile_.m)
        scores_max = self.extended_rule.base_rule(self.extended_rule.fake_profile_).scores_
        self.extended_rule.fake_profile_.scores[i] = np.zeros(self.profile_.m)
        scores_min = self.extended_rule.base_rule(self.extended_rule.fake_profile_).scores_
        self.extended_rule.fake_profile_.scores[i] = fake_scores_i

        all_scores = [(s, j, 1) for j, s in enumerate(scores_max)] \
                     + [(s, j, 0) for j, s in enumerate(scores_min)]

        all_scores.sort()
        all_scores = all_scores[::-1]


        best_manip = np.where(pref_ord == self.winner_)[0][0]
        for (_, j, kind) in all_scores:
            if kind == 0:
                break

            index_cand = np.where(pref_ord == j)[0][0]
            if index_cand < best_manip:
                k -= 1
                best_manip = index_cand
            unk -= 1

            if unk < 0:
                break

        best_manip = pref_ord[best_manip]

        if verbose:
            print(i, self.welfare_[best_manip])

        return best_manip

