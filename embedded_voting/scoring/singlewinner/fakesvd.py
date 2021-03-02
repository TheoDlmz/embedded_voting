# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringFunction


class FakeSVDRule(ScoringFunction):
    """
    Voting rule that apply the SVD method on another metrics than the matrix MM^t

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    similarity : function np.array, np.array -> float
        the similarity function between two voters' embeddings
    agg_rule: function np.array -> float
        the aggregation rule for singular values
    rc: boolean
        if True, use the square root of score in the matrix
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
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
