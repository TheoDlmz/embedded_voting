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

DROOP_QUOTA = 701
CLASSIC_QUOTA = 700
DROOP_QUOTA_MIN = 711
CLASSIC_QUOTA_MIN = 710

CANONICAL_BASIS = 801
SVD_BASIS = 802
SCORED_SVD_BASIS = 803


### Single-Winner scoring rules

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

    @cached_property
    def welfare_(self):
        scores = self.scores_
        max_score = np.max(scores)
        min_score = np.min(scores)
        if max_score == min_score:
            return np.ones(self.profile_.m)
        return (scores - min_score)/(max_score - min_score)

    def plot_winner(self):
        raise NotImplementedError


class RangeVoting(ScoringFunction):
    def score_(self, cand):
        return self.profile_.scores[::, cand].sum()


class Nash(ScoringFunction):
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

class ZonotopeRule(ScoringFunction):

    def __init__(self, profile=None, max_rank=None, power=1):
        self.max_rank = max_rank
        self.power = power
        super().__init__(profile=profile)

    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        if matrix_rank == 0:
            return (0, 0)
        if self.max_rank is not None:
            matrix_rank = min(self.max_rank, matrix_rank)
        #dim = self.profile_.dim
        volume = 0
        n = self.profile_.n
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n - matrix_rank:
            E = embeddings[current_subset, ...]
            if matrix_rank < self.profile_.dim:
                vol_i = np.linalg.det(np.dot(E, E.T))
                vol_i = max(0, vol_i)
                vol_i = np.sqrt(vol_i)
            else:
                vol_i = np.linalg.det(E)**self.power
                vol_i = np.abs(vol_i)
            volume += vol_i
            x = 1
            while current_subset[matrix_rank - x] == n - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return (matrix_rank, volume)

    @cached_property
    def ranking_(self):
        rank = [s[0] for s in self.scores_]
        scores = [s[1] for s in self.scores_]
        return np.lexsort((scores, rank))[::-1]

    def plot_winner(self, space="3D"):
        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=[self.winner_], list_titles=["ZonotopeRule Winner"])
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=[self.winner_], list_titles=["ZonotopeRule Winner"])
        else:
            raise ValueError("Incorrect space value (3D/2D)")


class maxCubeRule(ScoringFunction):

    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        if matrix_rank == 0:
            return (0, 0)

        # dim = self.profile_.dim
        volume = 0
        n = self.profile_.n
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n - matrix_rank:
            E = embeddings[current_subset, ...]
            volume = max(volume, np.sqrt(np.linalg.det(np.dot(E, E.T))))
            x = 1
            while current_subset[matrix_rank - x] == n - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return (matrix_rank, volume)

    @cached_property
    def ranking_(self):
        rank = [s[0] for s in self.scores_]
        scores = [s[1] for s in self.scores_]
        return np.lexsort((scores, rank))[::-1]

    def plot_winner(self, space="3D"):
        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=[self.winner_], list_titles=["MaxCubeRule Winner"])
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=[self.winner_], list_titles=["MaxCubeRule Winner"])
        else:
            raise ValueError("Incorrect space value (3D/2D)")

class SVDRule(ScoringFunction):
    def __init__(self, profile=None, agg_rule=np.prod, rc=False, use_rank=False, p=0):
        self.rc = rc
        self.agg_rule = agg_rule
        self.p = p
        self.use_rank = use_rank
        super().__init__(profile=profile)
    """
    def score_old_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand, rc=self.rc)
        X = self.profile_.embs
        prod_matrix = np.dot(X.T, X)
        prod_devant = np.linalg.matrix_power(prod_matrix, self.p)
        embeddings = np.dot(prod_devant, embeddings.T).T

        _, s, _ = np.linalg.svd(embeddings)

        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings)
            return (matrix_rank, self.agg_rule(s[:matrix_rank]))
        else:
            return self.agg_rule(s)
    """
    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand, rc=self.rc)
        X = self.profile_.embs
        prod_matrix = np.dot(X.T, X)
        prod_devant = np.linalg.matrix_power(prod_matrix, self.p)
        embeddings = np.dot(prod_devant, embeddings.T).T

        if (embeddings.shape[0] < embeddings.shape[1]):
            M_embeddings = embeddings.dot(embeddings.T)
        else:
            M_embeddings = (embeddings.T).dot(embeddings)

        s = np.linalg.eigvals(M_embeddings)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings)
            return (matrix_rank, self.agg_rule(s[:matrix_rank]))
        else:
            return self.agg_rule(s)

    @cached_property
    def ranking_(self):
        if self.use_rank:
            rank = [s[0] for s in self.scores_]
            scores = [s[1] for s in self.scores_]
            return np.lexsort((scores, rank))[::-1]
        else:
            return super().ranking_

    def set_rule(self, agg_rule):
        self.agg_rule = agg_rule
        self.delete_cache()
        return self

    def plot_winners(self, rule_list, rule_name, verbose=False, space="3D"):
        winners = []
        titles = []
        for (rule, name) in zip(rule_list, rule_name):
            self.set_rule(rule)
            if verbose:
                print("%s : %s" % (name, str(self.scores_)))
                print("Ranking : ", self.ranking_)
            winners.append(self.winner_)
            titles.append("Winner with SVD + %s" % name)

        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=winners, list_titles=titles)
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=winners, list_titles=titles)
        else:
            raise ValueError("Incorrect space value (3D/2D)")


class FeaturesRule(ScoringFunction):

    def __init__(self, profile=None, f=None, prod_score=True):
        if f is None:
            self.f = lambda u, v: np.dot(u, v)
        else:
            self.f = f
        self.prod_score = prod_score
        super().__init__(profile=profile)

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        try:
            return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T
        except:
            return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), S).T

    def features_quality(self):
        f = self.features_
        X = self.profile_.embs
        S = self.profile_.scores
        diff = np.dot(X, f.T) - S
        diff = diff ** 2
        diff = diff.mean(axis=0)
        return diff

    def score_(self, cand):
        score = 0
        for (v, s) in zip(self.profile_.embs, self.profile_.scores[::, cand]):
            temp = self.f(v, self.features_[cand])
            if self.prod_score:
                temp *= s
            score += temp
        return score

    def plot_winners(self, functions, functions_name, verbose=False, space="3D"):
        winners = []
        titles = []
        n_functions = len(functions)
        for i in range(n_functions):
            self.f = functions[i]
            self.delete_cache()
            if verbose:
                print("%s : %s" % (functions_name[i], str(self.scores_)))
                print("Ranking : ", self.ranking_)
            winners.append(self.winner_)
            titles.append("Winner with features + %s" % (functions_name[i]))

        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=winners, list_titles=titles)
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=winners, list_titles=titles)
        else:
            raise ValueError("Incorrect space value (3D/2D)")

    def plot_features(self, space="3D"):
        n_cand = self.profile_.m
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(30, n_rows * 5))
        intfig = [n_rows, 6, 1]
        f = self.features_
        for cand in range(n_cand):
            if space == "3D":
                ax = self.profile_._plot_scores_3D(self.profile_.scores[::, cand],
                                                   title="Candidate %i" % (cand + 1),
                                                   fig=fig,
                                                   intfig=intfig,
                                                   show=False)
                ax.plot([0, f[cand, 0]], [0, f[cand, 1]], [0, f[cand, 2]], color='k', linewidth=2)
                ax.scatter([f[cand, 0]], [f[cand, 1]], [f[cand, 2]], color='k', s=5)
            elif space == "2D":
                ax = self.profile_.plot_scores_2D(self.profile_.scores[::, cand],
                                                  title="Candidate %i" % (cand + 1),
                                                  show=False)

                fbis = np.maximum(f[cand], 0)
                fbis = fbis / np.linalg.norm(fbis)
                ax.scatter([fbis**2], color='k', s=50)
                plt.show()
            else:
                raise ValueError("Incorrect value for space (3D/2D)")

            intfig[2] += 1

        plt.show()


class SquareFeaturesRule(FeaturesRule):

    def score_(self, cand):
        return (self.features_[cand] ** 2).sum()

    def plot_winner(self, verbose=False, space="3D"):
        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=[self.winner_],
                                        list_titles=["Winner with sum of square of features"])
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=[self.winner_],
                                        list_titles=["Winner with sum of square of features"])
        else:
            raise ValueError("Incorrect space value (3D/2D)")


class ExtrapoleRule(ScoringFunction):

    def __init__(self, function, profile=None, basis=SVD_BASIS):
        self.function = function
        self.basis = basis
        super().__init__(profile)

    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand, rc=False)
        if self.basis == SCORED_SVD_BASIS:
            _, _, v = np.linalg.svd(embeddings, full_matrices=False)
            groups = np.dot(embeddings, v.T)
        elif self.basis == CANONICAL_BASIS:
            groups = embeddings
        elif self.basis == SVD_BASIS:
            _, _, v = np.linalg.svd(self.profile_.embs, full_matrices=False)
            groups = np.dot(embeddings, v.T)
        else:
            raise ValueError("Unknown Value for basis parameter")
        return self.function(groups)

    def plot_winners(self, functions, functions_name, verbose=False, space="3D"):
        winners = []
        titles = []
        n_functions = len(functions)
        for i in range(n_functions):
            self.function = functions[i]
            self.delete_cache()
            if verbose:
                print("%s : %s" % (functions_name[i], str(self.scores_)))
                print("Ranking : ", self.ranking_)
            winners.append(self.winner_)
            titles.append("Winner %s" % (functions_name[i]))

        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=winners, list_titles=titles)
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=winners, list_titles=titles)
        else:
            raise ValueError("Incorrect space value (3D/2D)")

    def plot_features(self):
        n_cand = self.profile_.m
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(30, n_rows * 5))
        intfig = [n_rows, 6, 1]
        for cand in range(n_cand):
            ax = self.profile_._plot_scores_3D(self.profile_.scores[::, cand],
                                               title="Candidate %i" % (cand + 1),
                                               fig=fig,
                                               intfig=intfig,
                                               show=False)
            if self.basis == CANONICAL_BASIS:
                features = np.eye(self.profile_.dim)
            elif self.basis == SVD_BASIS:
                _, _, v = np.linalg.svd(self.profile_.embs, full_matrices=False)
                features = v
            elif self.basis == SCORED_SVD_BASIS:
                embeddings = self.profile_.scored_embeddings(cand, rc=False)
                _, _, v = np.linalg.svd(embeddings, full_matrices=False)
                features = v
            else:
                raise ValueError("Unknown Value for basis parameter")
            for i in range(self.profile_.dim):
                if features[i, 0] < 0:
                    features[i] = - features[i]
                ax.plot([0, features[i, 0]], [0, features[i, 1]], [0, features[i, 2]], color='k', alpha=0.8,
                        linewidth=2)
                ax.scatter([features[i, 0]], [features[i, 1]], [features[i, 2]], color='k', alpha=0.8, s=5)
            intfig[2] += 1

        plt.show()



### Multiwinner Scoring rules

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

    def plot_winners_2D(self):
        winners, voters_list, vectors = self.ruleResults_
        ax = self.profile_.plot_profile_2D(show=False)
        for i, v in enumerate(vectors):
            v_temp = np.maximum(v, 0)
            v_temp = normalize(v_temp)
            ax.scatter([v_temp**2], color="k", alpha=0.8)
            #plt.text([v_temp**2], '#%i' % (i + 1), alpha=0.8)
        plt.show()

    def plot_voters(self):
        winners, voters_list, vectors = self.ruleResults_
        profile = self.profile_.embs
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
            ax = self.profile_._plot_scores_3D(self.profile_.scores[::, cand],
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
        return [self.profile_.scores[i, cand] * np.dot(self.profile_.embs[i], vec) for i in voters]


class IterFeatures(IterRules):

    def __init__(self, profile=None, k=None, f=None):
        if f is None:
            self.f = lambda u, v: np.dot(u, v)
        else:
            self.f = f
        super().__init__(profile=profile, k=k)

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if voters is None:
            voters = np.arange(self.profile_.n)
        scores = []
        features = self.compute_features(self.profile_.embs[voters], self.profile_.scores[voters])
        for cand in range(self.profile_.m):
            score = 0
            for (v, s) in zip(self.profile_.embs[voters], self.profile_.scores[voters, cand]):
                temp = self.f(v, features[cand])
                temp *= s
                score += temp
            scores.append(score)
        return scores, features

    def satisfaction(self, winner_j, vec, voters):
        temp = [self.f(self.profile_.embs[i], vec) for i in voters]
        temp = [self.profile_.scores[v, winner_j] * temp[i] for i, v in enumerate(voters)]
        return temp


class SquareIterFeatures(IterRules):

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if voters is None:
            voters = np.arange(self.profile_.n)
        features = self.compute_features(self.profile_.embs[voters], self.profile_.scores[voters])
        scores = np.sum(features ** 2, axis=1)
        return scores, features

    def satisfaction(self, winner_j, vec, voters):
        temp = [np.dot(self.profile_.embs[i], vec) for i in voters]
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
        fig = plt.figure(figsize=(60, n_rows * 10))
        intfig = [n_rows, 6, 1]
        for i in range(n_cand):
            _ = self.profile_._plot_scores_3D(self.ls_weight[i],
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
        return [self.profile_.scores[i, cand] * np.dot(self.profile_.embs[i], vec) for i in range(self.profile_.n)]


class WeightedIterFeatures(WeightedIterRules):

    def __init__(self, profile=None, k=None, quota=CLASSIC_QUOTA, f=None):
        if f is None:
            self.f = lambda u, v: np.dot(u, v)
        else:
            self.f = f
        super().__init__(profile=profile, k=k, quota=quota)

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        scores = []
        features = self.compute_features(self.profile_.embs, np.dot(np.diag(self.weights), self.profile_.scores))
        for cand in range(self.profile_.m):
            score = 0
            for (v, s, w) in zip(self.profile_.embs, self.profile_.scores[::, cand], self.weights):
                temp = self.f(v, features[cand])
                temp *= s * w
                score += temp
            scores.append(score)
        return scores, features

    def satisfaction(self, winner_j, vec, voters=None):

        temp = [np.dot(self.profile_.embs[i], vec) for i in range(self.profile_.n)]
        temp = [self.profile_.scores[i, winner_j] * temp[i] for i in range(self.profile_.n)]

        return temp


class WeightedSquareIterFeatures(WeightedIterRules):

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    @staticmethod
    def compute_features(X, S):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T

    def winner_k(self, voters=None):
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n)

        features = self.compute_features(self.profile_.embs[voters],
                                         np.dot(np.diag(self.weights), self.profile_.scores[voters]))
        scores = np.sum(features ** 2, axis=1)
        return scores, features

    def satisfaction(self, winner_j, vec, voters=None):
        temp = [np.dot(self.profile_.embs[i], vec) for i in range(self.profile_.n)]
        temp = [self.profile_.scores[i, winner_j] * temp[i] for i in range(self.profile_.n)]
        return temp

class PositionalRuleExtension(ScoringFunction):

    def __init__(self, profile=None,  points=None, rule=None):
        self.profile_ = profile
        if len(points) != self.profile_.m:
            raise ValueError("The positional rule must be of length %i" % (self.profile_.m))
        self.points = points
        if rule != None:
            self.base_rule = rule
        else:
            self.base_rule = None
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = None

    def __call__(self, profile):
        self.profile_ = profile
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = self.base_rule(self.fake_profile_)
        self.delete_cache()
        return self

    def set_rule(self, rule):
        self.base_rule = rule
        self.fake_profile_ = self.create_fake_profile()
        self.rule_ = self.base_rule(self.fake_profile_)
        self.delete_cache()
        return self

    def create_fake_profile(self):
        points = np.array(self.points)/np.max(self.points)
        fake_profile = np.zeros((self.profile_.n, self.profile_.m))
        for i in range(self.profile_.n):
            scores_i = self.profile_.scores[i]
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = Profile(self.profile_.m, self.profile_.dim)
        self.profile_.copy(p)
        p.scores = fake_profile
        return p

    def score_(self, cand):
        return self.rule_.scores_[cand]

    def plot_fake_profile(self):
        self.fake_profile_.plot_cands_3D()


class PluralityExtension(PositionalRuleExtension):

    def __init__(self, profile, rule=None):
        m = profile.m
        points = [1] + [0]*(m-1)
        super().__init__(profile, points, rule)


class kApprovalExtension(PositionalRuleExtension):

    def __init__(self, profile, k=2, rule=None):
        m = profile.m
        assert(k < m)
        points = [1]*k + [0]*(m-k)
        super().__init__(profile, points, rule)


class VetoExtension(PositionalRuleExtension):

    def __init__(self, profile, rule=None):
        m = profile.m
        points = [1]*(m-1) + [0]
        super().__init__(profile, points, rule)

class BordaExtension(PositionalRuleExtension):

    def __init__(self, profile, rule=None):
        m = profile.m
        points = [m-i-1 for i in range(m)]
        super().__init__(profile, points, rule)

class InstantRunoffExtension(ScoringFunction):
    def __init__(self, profile=None,  rule=None):
        self.profile_ = profile
        self.rule_ = rule

    def __call__(self, profile):
        self.profile_ = profile
        self.delete_cache()
        return self

    def set_rule(self, rule):
        self.rule_ = rule
        self.delete_cache()
        return self

    @cached_property
    def ranking_(self):
        m = self.profile_.m
        ranking = np.zeros(m, dtype=int)
        eliminated = []
        for i in range(m):
            fake_profile = self.create_fake_profile(eliminated)
            rule_i = self.rule_(fake_profile)
            loser = rule_i.ranking_[m-1-i]
            ranking[m-i-1] = loser
            eliminated.append(loser)
        return ranking

    def create_fake_profile(self, eliminated):
        fake_profile = np.zeros((self.profile_.n, self.profile_.m))
        points = np.zeros(self.profile_.m)
        points[0] = 1

        for i in range(self.profile_.n):
            scores_i = self.profile_.scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        p = Profile(self.profile_.m, self.profile_.dim)
        self.profile_.copy(p)
        p.scores = fake_profile
        return p



class FakeSVDRule(ScoringFunction):
    def __init__(self,  profile=None, similarity=None, agg_rule=np.prod, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = agg_rule
        self.use_rank = use_rank
        self.similarity = similarity
        super().__init__(profile=profile)

    def score_(self, cand):
        M_embeddings = self.profile_.fake_covariance_matrix(cand, self.similarity, rc=self.rc)

        s = np.linalg.eigvals(M_embeddings)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(M_embeddings)
            return (matrix_rank, self.agg_rule(s[:matrix_rank]))
        else:
            return self.agg_rule(s)

    @cached_property
    def ranking_(self):
        if self.use_rank:
            rank = [s[0] for s in self.scores_]
            scores = [s[1] for s in self.scores_]
            return np.lexsort((scores, rank))[::-1]
        else:
            return super().ranking_

    def set_rule(self, agg_rule):
        self.agg_rule = agg_rule
        self.delete_cache()
        return self
