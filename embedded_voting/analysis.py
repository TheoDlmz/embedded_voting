# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.profile.Profile import Profile
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.utils.plots import create_3D_plot
import itertools


class MovingVoter(DeleteCacheMixin):

    def __init__(self, r=None):
        self.rule_ = None
        self.profile_ = Profile(4, 3)
        self.profile_.add_group(1, [0, 0, 1], [0, 0, 1, 0.5], 0, 0, 'red')
        self.profile_.add_group(1, [0, 1, 0], [0, 1, 0, 0.5], 0, 0, 'blue')
        self.profile_.add_group(1, [1, 0, 0], [1, 0, 0, 0.5], 0, 0, 'green')
        self.profile_.add_group(1, [1, 0, 0], [0.75, 0.75, 0.75, 0.75], 0, 0, 'orange')
        if r is not None:
            self(r)

    def __call__(self, r):
        self.rule_ = r
        return self

    def plot_evol(self):
        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.profile_.profile[3] = normalize([1-x, x, 0])
            tab_y.append(self.rule_(self.profile_).scores_)

        tab_y = np.array(tab_y).T
        name = ["Start", "End", "Orth", "Consensus"]
        for i in range(4):
            plt.plot(tab_x, tab_y[i], label=name[i])

        plt.title("Evolution of the scoring")
        plt.xlabel("x coordinate of moving voter")
        plt.ylabel("Score")
        plt.xlim(0, 1)
        #plt.ylim(0)
        plt.legend()
        plt.show()




class MovingVoterFeatures(DeleteCacheMixin):
    def __init__(self, r=None):
        self.rule_ = None
        self.profile_ = Profile(4, 3)
        self.profile_.add_group(1, [0, 0, 1], [0, 0, 1, 0.5], 0, 0, 'red')
        self.profile_.add_group(1, [0, 1, 0], [0, 1, 0, 0.5], 0, 0, 'blue')
        self.profile_.add_group(1, [1, 0, 0], [1, 0, 0, 0.5], 0, 0, 'green')
        self.profile_.add_group(1, [1, 0, 0], [0.75, 0.75, 0.75, 0.75], 0, 0, 'orange')
        if r is not None:
            self(r)

    def __call__(self, r):
        self.rule_ = r
        return self

    def plot_evol(self):
        tab_x = np.linspace(0, 1, 20)
        tab_y = []
        for x in tab_x:
            self.profile_.profile[3] = normalize([1 - x, x, 0])
            _, vectors = self.rule_(self.profile_).winner_k()
            tab_y.append(vectors)
        tab_y = np.array(tab_y)

        fig = plt.figure(figsize=(10, 10))
        ax = create_3D_plot(fig)
        colors = ["blue", "orange", "green", "red"]
        name = ["Start", "End", "Orth", "Consensus"]
        for i in range(4):
            ax.plot(tab_y[::, i, 0], tab_y[::, i, 1], tab_y[::, i, 2], color=colors[i], alpha=0.5, label=name[i])
            for j, v in enumerate(tab_y[::, i]):
                ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=colors[i], alpha=j/60)
        plt.legend()
        plt.show()



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
                scores_max = self.extended_rule._rule(self.profile_).scores_

                fake_profile[i] = np.zeros(self.profile_.m)
                self.profile_.scores = fake_profile
                scores_min = self.extended_rule._rule(self.profile_).scores_

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


class ManipulationCoalitionExtension(ManipulationCoalition):

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

    def manipulation_triviale(self, cand, verbose=False):

        voters_interested = []
        for i in range(self.profile_.n):
            score_i = self.profile_.scores[i]
            if score_i[self.winner_] < score_i[cand]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i" % (len(voters_interested), cand, self.winner_))

        old_profile = self.profile_.scores.copy()
        for i in voters_interested:
            self.profile_.scores[i][self.winner_] = -1
            self.profile_.scores[i][cand] = 2

        new_winner = self.extended_rule(self.profile_).winner_
        self.profile_.scores = old_profile

        if verbose:
            print("Winner is %i" % new_winner)

        return (new_winner == cand)
