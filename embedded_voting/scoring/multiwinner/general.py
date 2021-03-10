"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.utils.miscellaneous import normalize
import matplotlib.pyplot as plt

DROOP_QUOTA = 701
CLASSIC_QUOTA = 700
DROOP_QUOTA_MIN = 711
CLASSIC_QUOTA_MIN = 710


class MultiwinnerRule(DeleteCacheMixin):
    """
    A class for rules that elect a committee of candidates

    Parameters
    __________
    profile : Profile
        the profile of voters
    k : int
        the size of the committee
    """
    def __init__(self, profile=None, k=None):
        self.profile_ = None
        self.k_ = None
        if (profile is not None) and (k is not None):
            self(profile, k)

    def __call__(self, profile, k=1):
        self.profile_ = profile
        self.k_ = k
        self.delete_cache()
        return self

    def set_k(self, k):
        self.delete_cache()
        self.k_ = k

    @cached_property
    def winners_(self):
        raise NotImplementedError


class IterRules(MultiwinnerRule):
    """
    A class for iteratives multi winner rules

    Parameters
    __________
    profile : Profile
        the profile of voters
    k : int
        the size of the committee
    quota : {DROOP_QUOTA, CLASSIC_QUOTA, DROOP_QUOTA_MIN, CLASSIC_QUOTA_MIN}
        the quota used for the re-weighing step
    """
    def __init__(self, profile=None, k=None, quota=CLASSIC_QUOTA):
        self.quota = quota
        self.weights = np.ones(0)
        super().__init__(profile=profile, k=k)

    def winner_k(self):
        raise NotImplementedError

    def satisfaction(self, cand, vec):
        raise NotImplementedError

    def updateWeight(self, sat, j):
        if self.quota in [CLASSIC_QUOTA, CLASSIC_QUOTA_MIN]:
            quota_val = self.profile_.n / self.k_
        elif self.quota in [DROOP_QUOTA, DROOP_QUOTA_MIN]:
            quota_val = self.profile_.n / (self.k_ + 1) + 1
        else:
            raise ValueError("Quota Not Existing")

        temp = [sat[i] * self.weights[i] for i in range(self.profile_.n)]
        total_sat = np.sum(temp)

        if self.quota in [CLASSIC_QUOTA_MIN, DROOP_QUOTA_MIN]:
            quota_val = min(quota_val, total_sat)

        pond_weights = np.array(temp) * quota_val / total_sat
        self.weights = np.maximum(0, self.weights - pond_weights)

    @cached_property
    def ruleResults_(self):
        n_voters = self.profile_.n

        frac = n_voters // self.k_

        winners = []
        voters_list = []
        voters = np.arange(n_voters)
        vectors = []
        self.weights = np.ones(n_voters)
        ls_weights = [self.weights]

        for j in range(self.k_):
            scores, vectors_scores = self.winner_k()
            scores = np.array(scores)
            scores[winners] = 0

            winner_j = np.argmax(scores)
            vec = vectors_scores[winner_j]
            vectors.append(vec)
            winners.append(winner_j)

            satisfactions = self.satisfaction(winner_j, vec)

            self.updateWeight(satisfactions, j)
            ls_weights.append(self.weights)

            sorted_sat = np.argsort(satisfactions)[::-1]
            voters_list.append(voters[sorted_sat[:frac:]])

        self.ls_weight = ls_weights

        return winners, voters_list, vectors

    def plot_weights(self):
        _ = self.ruleResults_
        n_cand = len(self.ls_weight)
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(60, n_rows * 10))
        intfig = [n_rows, 6, 1]
        for i in range(n_cand):
            _ = self.profile_.plot_scores_3D(self.ls_weight[i],
                                             title="Step %i" % i,
                                             fig=fig,
                                             intfig=intfig,
                                             show=False)

            intfig[2] += 1

        sum_w = [self.ls_weight[i].sum() / (n_cand - i - 1) for i in range(n_cand - 1)]
        print("Weight / remaining candidate : ", sum_w)
        plt.show()

    @cached_property
    def winners_(self):
        return self.ruleResults_[0]

    def plot_winners(self):
        winners = self.winners_
        titles = ["Winner n°%i" % (i + 1) for i in range(self.k_)]

        self.profile_.plot_cands_3D(winners, titles)

    def plot_winners_2D(self):
        winners, voters_list, vectors = self.ruleResults_
        ax = self.profile_.plot_profile_2D(show=False)
        for i, v in enumerate(vectors):
            v_temp = np.maximum(v, 0)
            v_temp = normalize(v_temp)
            ax.scatter([v_temp**2], color="k", alpha=0.8)
        plt.show()

    def plot_vectors(self):
        n_cand = self.profile_.m
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(30, n_rows * 5))
        intfig = [n_rows, 6, 1]
        _, vectors = self.winner_k()
        for cand in range(n_cand):
            ax = self.profile_.plot_scores_3D(self.profile_.scores[::, cand],
                                              title="Candidate %i" % (cand + 1),
                                              fig=fig,
                                              intfig=intfig,
                                              show=False)

            ax.plot([0, vectors[cand][0]], [0, vectors[cand][1]], [0, vectors[cand][2]], color='k', linewidth=2)
            ax.scatter([vectors[cand][0]], [vectors[cand][1]], [vectors[cand][2]], color='k', s=5)
            intfig[2] += 1

        plt.show()
