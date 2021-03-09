# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule


class SVDRule(ScoringRule):
    """
    Voting rule based on singular values of the embedding matrix

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values. Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix. Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, aggregation_rule=np.prod, square_root=True, use_rank=False):
        self.square_root = square_root
        self.aggregation_rule = aggregation_rule
        self.use_rank = use_rank
        if use_rank:
            self.score_components = 2
        else:
            self.score_components = 1
        super().__init__(profile=profile)

    def score_(self, candidate):
        embeddings = self.profile_.scored_embeddings(candidate, square_root=self.square_root)

        if embeddings.shape[0] < embeddings.shape[1]:
            embeddings_matrix = embeddings.dot(embeddings.T)
        else:
            embeddings_matrix = embeddings.T.dot(embeddings)

        s = np.linalg.eigvals(embeddings_matrix)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings)
            return matrix_rank, self.aggregation_rule(s[:matrix_rank])
        else:
            return self.aggregation_rule(s)

    def set_rule(self, agg_rule):
        self.aggregation_rule = agg_rule
        self.delete_cache()
        return self


class SVDNash(SVDRule):
    """
    Voting rule based on the product of the singular values of the embedding matrix

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values. Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix. Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, square_root=False, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.prod, square_root=square_root, use_rank=use_rank)


class SVDSum(SVDRule):
    """
    Voting rule based on the sum of the singular values of the embedding matrix

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values. Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix. Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, square_root=False, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.sum, square_root=square_root, use_rank=use_rank)


class SVDMin(SVDRule):
    """
    Voting rule based on the minimum of the singular values of the embedding matrix

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values. Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix. Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, square_root=False, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.min, square_root=square_root, use_rank=use_rank)


class SVDMax(SVDRule):
    """
    Voting rule based on the maximum of the singular values of the embedding matrix

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values. Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix. Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, square_root=False, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.max, square_root=square_root, use_rank=use_rank)


class SVDLog(SVDRule):
    """
    Voting rule based on the sum of the log of 1 + the singular values of the embedding matrix divided
    by some constant.

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election
    const : float > 0
        the constant in the log function
    aggregation_rule: function np.array -> float
        the aggregation rule for singular values. Default is product.
    square_root: boolean
        if True, use the square root of score in the matrix. Default is True.
    use_rank : boolean
        if True, consider the rank of the matrix when doing the ranking

    """
    def __init__(self, profile=None, const=1, square_root=False, use_rank=False):
        sum_log = lambda x: np.sum(np.log(1+x/const))
        super().__init__(profile=profile, aggregation_rule=sum_log, square_root=square_root, use_rank=use_rank)
