import numpy as np
from embedded_voting.manipulation.voter.general import ManipulationExtension


class ManipulationIRV(ManipulationExtension):

    def create_fake_scores(self, eliminated, scores):
        fake_profile = np.zeros((self.profile_.n, self.profile_.m))
        points = np.zeros(self.profile_.m)
        points[0] = 1

        for i in range(self.profile_.n):
            scores_i = scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        return fake_profile

    def manipulation_voter(self, i, verbose=False):
        scores = self.profile_.scores.copy()
        score_i = self.profile_.scores[i].copy()
        pref_ord = np.argsort(score_i)[::-1]

        m = self.profile_.m

        if pref_ord[0] == self.winner_:
            return self.winner_


        best_manip_i = np.where(pref_ord == self.winner_)[0][0]

        queue_eliminated = [([],-1)]

        while len(queue_eliminated) > 0:
            (el, one) = queue_eliminated.pop()
            #print(el)

            if len(el) == m:
                winner = el[-1]
                index_cand = np.where(pref_ord == winner)[0][0]
                if index_cand < best_manip_i:
                    best_manip_i = index_cand
            else:
                fake_profile = self.create_fake_scores(el, scores)
                fake_profile[i] = np.ones(self.profile_.m)
                self.profile_.scores = fake_profile
                scores_max = self.extended_rule.rule_(self.profile_).scores_

                fake_profile[i] = np.zeros(self.profile_.m)
                self.profile_.scores = fake_profile
                scores_min = self.extended_rule.rule_(self.profile_).scores_

                all_scores = [(s, j, 1) for j, s in enumerate(scores_max) if j not in el] \
                             + [(s, j, 0) for j, s in enumerate(scores_min) if j not in el]

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


        best_manip = pref_ord[best_manip_i]

        if verbose:
            print(self.welfare_[best_manip])

        return best_manip
