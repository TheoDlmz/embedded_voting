# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule


class FakeSVDRule(ScoringRule):
    """
    Voting rule that apply the SVD method on another matrix than the embeddings matrix.

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    similarity : function np.array, np.array -> float
        the similarity function between two voters' embeddings
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values.
        Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix.
        Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking.
        Default is false

    """
    def __init__(self,  profile=None, similarity=None, aggregation_rule=np.prod, square_root=True, use_rank=False):
        super().__init__(profile=profile)
        self.square_root = square_root
        self.aggregation_rule = aggregation_rule
        self.use_rank = use_rank
        self.similarity = similarity
        if use_rank:
            self.score_components = 2

    def set_rule(self, aggregation_rule):
        self.aggregation_rule = aggregation_rule
        self.delete_cache()
        return self

    def score_(self, candidates):
        embeddings_matrix = self.profile_.fake_covariance_matrix(candidates,
                                                                 self.similarity,
                                                                 square_root=self.square_root)
        s = np.linalg.eigvals(embeddings_matrix)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings_matrix)
            return matrix_rank, self.aggregation_rule(s[:matrix_rank])
        else:
            return self.aggregation_rule(s)
