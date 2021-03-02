# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringFunction


class SVDRule(ScoringFunction):
    """
    Voting rule based on singular values of the matrix

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    agg_rule: function np.array -> float
        the aggregation rule for singular values
    rc: boolean
        if True, use the square root of score in the matrix
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, agg_rule=np.prod, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = agg_rule
        self.use_rank = use_rank
        super().__init__(profile=profile)

    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand, rc=self.rc)

        if embeddings.shape[0] < embeddings.shape[1]:
            M_embeddings = embeddings.dot(embeddings.T)
        else:
            M_embeddings = embeddings.T.dot(embeddings)

        s = np.linalg.eigvals(M_embeddings)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings)
            return matrix_rank, self.agg_rule(s[:matrix_rank])
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



class SVDNash(SVDRule):
    def __init__(self, profile=None, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = np.prod
        self.use_rank = use_rank
        super().__init__(profile=profile)


class SVDSum(SVDRule):
    def __init__(self, profile=None, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = np.sum
        self.use_rank = use_rank
        super().__init__(profile=profile)


class SVDMin(SVDRule):
    def __init__(self, profile=None, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = np.min
        self.use_rank = use_rank
        super().__init__(profile=profile)


class SVDMax(SVDRule):
    def __init__(self, profile=None, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = np.max
        self.use_rank = use_rank
        super().__init__(profile=profile)


class SVDLog(SVDRule):
    def __init__(self, profile=None, c=1, rc=False, use_rank=False):
        self.rc = rc
        self.agg_rule = lambda x: np.sum(np.log(1+x/c))
        self.use_rank = use_rank
        super().__init__(profile=profile)
