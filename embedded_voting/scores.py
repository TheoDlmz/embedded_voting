# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from embedded_voting.utils import DeleteCacheMixin, cached_property, create_3D_plot

DROOP_QUOTA = 701
CLASSIC_QUOTA = 700
DROOP_QUOTA_MIN = 711
CLASSIC_QUOTA_MIN = 710

class ScoringFunction(DeleteCacheMixin):

    def __init__(self, profile=None):
        self.profile_ = None
        if profile is not None:
            self(profile)

    def __call__(self, profile):
        self.profile_ = profile
        self.delete_cache()
        return self

    def score_(self, cand):
        raise NotImplementedError

    @cached_property
    def scores_(self):
        return [self.score_(cand) for cand in range(self.profile_.m)]

    @cached_property
    def ranking_(self):
        return np.argsort(self.scores_)[::-1]

    @cached_property
    def winner_(self):
        return self.ranking_[0]

    def plot_winner(self):
        raise NotImplementedError


class ZonotopeRule(ScoringFunction):

    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand)
        dim = self.profile_.dim
        volume = 0
        n = self.profile_.n
        current_subset = list(np.arange(dim))
        while current_subset[0] <= n - dim:
            E = embeddings[current_subset, ...]
            volume += np.abs(np.linalg.det(E))
            x = 1
            while current_subset[dim - x] == n - x:
                x += 1
            val = current_subset[dim - x] + 1
            while x > 0:
                current_subset[dim - x] = val
                val += 1
                x -= 1

        return volume

    def plot_winner(self):
        self.profile_.plot_cands_3D(list_cand=[self.winner_], list_titles=["ZonotopeRule Winner"])


class SVDRule(ScoringFunction):

    def __init__(self, profile=None, agg_rule=np.prod, rc=False):
        self.rc = rc
        self.agg_rule = agg_rule
        super().__init__(profile=profile)

    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand, rc=self.rc)
        _, s, _ = np.linalg.svd(embeddings, full_matrices=False)

        return self.agg_rule(s)

    def set_rule(self, agg_rule):
        self.agg_rule = agg_rule
        self.delete_cache()
        return self

    def plot_winners(self, rule_list, rule_name, verbose=False):
        winners = []
        titles = []
        for (rule, name) in zip(rule_list, rule_name):
            self.set_rule(rule)
            if verbose:
                print("%s : %s" % (name, str(self.scores_)))
                print("Ranking : ", self.ranking_)
            winners.append(self.winner_)
            titles.append("Winner with SVD + %s" % name)

        self.profile_.plot_cands_3D(list_cand=winners, list_titles=titles)


class FeaturesRule(ScoringFunction):

    def __init__(self, profile=None, log=False, sin=True, prod_score=True):
        self.log = log
        self.sin = sin
        self.prod_score = prod_score
        super().__init__(profile=profile)

    @cached_property
    def features_(self):
        X = self.profile_.profile
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def features_quality(self):
        f = self.features_
        X = self.profile_.profile
        S = self.profile_.scores
        diff = np.dot(X, f.T) - S
        diff = diff ** 2
        diff = diff.mean(axis=0)
        return diff

    def score_(self, cand):
        score = 0
        for (v, s) in zip(self.profile_.profile, self.profile_.scores[::, cand]):
            temp = np.dot(v, self.features_[cand])
            if self.sin:
                temp = 1 - temp
            if self.prod_score:
                temp *= s
            if self.log:
                temp = np.log(1 + temp)
            score += temp
        return score

    def set_log(self, log):
        self.delete_cache()
        self.log = log

    def set_sin(self, sin):
        self.delete_cache()
        self.sin = sin

    def plot_winners(self, verbose=False):
        winners = []
        titles = []
        for sin in [False, True]:
            for log in [False, True]:
                self.set_sin(sin)
                self.set_log(log)
                if sin:
                    sinstr = "Sinus"
                else:
                    sinstr = "Cosinus"
                logstr = ""
                if log:
                    logstr = "+ Log"
                if verbose:
                    print("%s %s : %s" % (sinstr, logstr, str(self.scores_)))
                    print("Ranking : ", self.ranking_)
                winners.append(self.winner_)
                titles.append("Winner with features + %s %s" % (sinstr, logstr))

        self.profile_.plot_cands_3D(list_cand=winners, list_titles=titles)

    def plot_features(self):

        n_cand = self.profile_.m
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(30, n_rows * 5))
        intfig = [n_rows, 6, 1]
        f = self.features_
        for cand in range(n_cand):
            ax = self.profile_.plot_scores_3D(self.profile_.scores[::, cand],
                                              title="Candidate %i" % (cand + 1),
                                              fig=fig,
                                              intfig=intfig,
                                              show=False)

            ax.plot([0, f[cand, 0]], [0, f[cand, 1]], [0, f[cand, 2]], color='k', linewidth=2)
            ax.scatter([f[cand, 0]], [f[cand, 1]], [f[cand, 2]], color='k', s=5)
            intfig[2] += 1

        plt.show()


class SquareFeaturesRule(FeaturesRule):

    def score_(self, cand):
        return (self.features_[cand] ** 2).sum()

    def plot_winner(self, verbose=False):
        self.profile_.plot_cands_3D(list_cand=[self.winner_],
                                    list_titles=["Winner with sum of square of features"])


class MultiwinnerRules(DeleteCacheMixin):
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


class IterRules(MultiwinnerRules):

    def winner_k(self, voters):
        raise NotImplementedError

    def satisfaction(self, cand, vec, voters):
        raise NotImplementedError

    @cached_property
    def ruleResults_(self):
        n_voters = self.profile_.n

        frac = n_voters // self.k_

        winners = []
        voters_list = []
        voters = np.arange(n_voters)
        vectors = []
        for j in range(self.k_):
            scores, vectors_scores = self.winner_k(voters)
            scores = np.array(scores)
            scores[winners] = 0

            winner_j = np.argmax(scores)
            vec = vectors_scores[winner_j]
            vectors.append(vec)
            winners.append(winner_j)

            satisfactions = self.satisfaction(winner_j, vec, voters)

            sorted_sat = np.argsort(satisfactions)[::-1]
            voters_list.append(voters[sorted_sat[:frac:]])
            voters = voters[sorted_sat[frac::]]

        return winners, voters_list, vectors

    @cached_property
    def winners_(self):
        return self.ruleResults_[0]

    def plot_winners(self):
        winners = self.winners_
        titles = ["Winner n°%i" % (i + 1) for i in range(self.k_)]

        self.profile_.plot_cands_3D(winners, titles)

    def plot_voters(self):
        winners, voters_list, vectors = self.ruleResults_
        profile = self.profile_.profile
        scores = self.profile_.scores
        n_rows = (self.k_ - 1) // 6 + 1
        fig = plt.figure(figsize=(30, 5 * n_rows))
        intfig = [n_rows, 6, 1]
        for j in range(self.k_):
            ax = create_3D_plot(fig, intfig)
            for i, (v, s) in enumerate(zip(profile[voters_list[j]], scores[voters_list[j], winners[j]])):
                g = self.profile_.groups[voters_list[j][i]]
                ax.plot([0, s * v[0]], [0, s * v[1]], [0, s * v[2]], color=self.profile_.color_groups[g], alpha=0.3)
            ax.plot([0, vectors[j][0]], [0, vectors[j][1]], [0, vectors[j][2]], color='k', linewidth=2)
            ax.scatter([vectors[j][0]], [vectors[j][1]], [vectors[j][2]], color='k', s=5)
            ax.set_title("Winner n°%i" % (j + 1), fontsize=24)
            intfig[2] += 1
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


class IterSVD(IterRules):

    def __init__(self, profile=None, k=None, agg_rule=np.max):
        self.agg_rule = agg_rule
        super().__init__(profile=profile, k=k)

    def winner_k(self, voters=None):
        if voters is None:
            voters = np.arange(self.profile_.n)
        vectors = []
        scores = []
        for cand in range(self.profile_.m):
            X = self.profile_.scored_embeddings(cand)[voters]
            _, s, v = np.linalg.svd(X, full_matrices=False)
            scores.append(self.agg_rule(s))
            if (v[0] <= 0).all():
                v[0] = -v[0]
            vectors.append(v[0])
        return scores, vectors

    def satisfaction(self, cand, vec, voters):
        return [self.profile_.scores[i, cand] * np.dot(self.profile_.profile[i], vec) for i in voters]


class IterFeatures(IterRules):

    def __init__(self, profile=None, k=None, log=False, sin=False):
        self.log = log
        self.sin = sin
        super().__init__(profile=profile, k=k)

    @cached_property
    def features_(self):
        X = self.profile_.profile
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if voters is None:
            voters = np.arange(self.profile_.n)
        scores = []
        features = self.compute_features(self.profile_.profile[voters], self.profile_.scores[voters])
        for cand in range(self.profile_.m):
            score = 0
            for (v, s) in zip(self.profile_.profile[voters], self.profile_.scores[voters, cand]):
                temp = np.dot(v, features[cand])
                if self.sin:
                    temp = 1 - temp
                temp *= s
                if self.log:
                    temp = np.log(1 + temp)
                score += temp
            scores.append(score)
        return scores, features

    def satisfaction(self, winner_j, vec, voters):
        temp = [np.dot(self.profile_.profile[i], vec) for i in voters]
        if self.sin:
            temp = [1 - x for x in temp]
        temp = [self.profile_.scores[v, winner_j] * temp[i] for i, v in enumerate(voters)]
        if self.log:
            temp = [np.log(x) for x in temp]

        return temp


class SquareIterFeatures(IterRules):

    @cached_property
    def features_(self):
        X = self.profile_.profile
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if voters is None:
            voters = np.arange(self.profile_.n)
        features = self.compute_features(self.profile_.profile[voters], self.profile_.scores[voters])
        scores = np.sum(features ** 2, axis=1)
        return scores, features

    def satisfaction(self, winner_j, vec, voters):
        temp = [np.dot(self.profile_.profile[i], vec) for i in voters]
        temp = [self.profile_.scores[v, winner_j] * temp[i] for i, v in enumerate(voters)]
        return temp


class WeightedIterRules(IterRules):

    def __init__(self, profile=None, k=None, quota=CLASSIC_QUOTA):
        self.quota = quota
        self.weights = np.ones(0)
        super().__init__(profile=profile, k=k)

    def winner_k(self, voters):
        raise NotImplementedError

    def satisfaction(self, cand, vec, voters):
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
            scores, vectors_scores = self.winner_k(voters)
            scores = np.array(scores)
            scores[winners] = 0

            winner_j = np.argmax(scores)
            vec = vectors_scores[winner_j]
            vectors.append(vec)
            winners.append(winner_j)

            satisfactions = self.satisfaction(winner_j, vec, voters)

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
        fig = plt.figure(figsize=(30, n_rows * 5))
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


class WeightedIterSVD(WeightedIterRules):

    def __init__(self, profile=None, k=None, quota=CLASSIC_QUOTA, agg_rule=np.max):
        self.agg_rule = agg_rule
        super().__init__(profile=profile, k=k, quota=quota)

    def winner_k(self, voters=None):
        vectors = []
        scores = []
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        for cand in range(self.profile_.m):
            X = np.dot(np.diag(self.weights), self.profile_.scored_embeddings(cand))
            _, s, v = np.linalg.svd(X, full_matrices=False)
            scores.append(self.agg_rule(s))
            if (v[0] <= 0).all():
                v[0] = -v[0]
            vectors.append(v[0])
        return scores, vectors

    def satisfaction(self, cand, vec, voters=None):
        return [self.profile_.scores[i, cand] * np.dot(self.profile_.profile[i], vec) for i in range(self.profile_.n)]


class WeightedIterFeatures(WeightedIterRules):

    def __init__(self, profile=None, k=None, quota=CLASSIC_QUOTA, log=False, sin=False):
        self.log = log
        self.sin = sin
        super().__init__(profile=profile, k=k, quota=quota)

    @cached_property
    def features_(self):
        X = self.profile_.profile
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        scores = []
        features = self.compute_features(self.profile_.profile, np.dot(np.diag(self.weights), self.profile_.scores))
        for cand in range(self.profile_.m):
            score = 0
            for (v, s, w) in zip(self.profile_.profile, self.profile_.scores[::, cand], self.weights):
                temp = np.dot(v, features[cand])
                if self.sin:
                    temp = 1 - temp
                temp *= s * w
                if self.log:
                    temp = np.log(1 + temp)
                score += temp
            scores.append(score)
        return scores, features

    def satisfaction(self, winner_j, vec, voters=None):

        temp = [np.dot(self.profile_.profile[i], vec) for i in range(self.profile_.n)]
        if self.sin:
            temp = [1 - x for x in temp]
        temp = [self.profile_.scores[i, winner_j] * temp[i] for i in range(self.profile_.n)]

        return temp


class WeightedSquareIterFeatures(WeightedIterRules):

    @cached_property
    def features_(self):
        X = self.profile_.profile
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        features = self.compute_features(self.profile_.profile[voters],
                                         np.dot(np.diag(self.weights), self.profile_.scores[voters]))
        scores = np.sum(features ** 2, axis=1)
        return scores, features

    def satisfaction(self, winner_j, vec, voters=None):
        temp = [np.dot(self.profile_.profile[i], vec) for i in range(self.profile_.n)]
        temp = [self.profile_.scores[i, winner_j] * temp[i] for i in range(self.profile_.n)]
        return temp
