# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile


class FakeSVDRule(ScoringRule):
    """
    Voting rule that apply the SVD method
    on another matrix than the embeddings matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    similarity : callable
        A similarity function that associate a pair
        of feature vectors to a similarity value (a float).
        Input : np.ndarray, np.ndarray.
        Output : float.
    aggregation_rule: callable
        The aggregation rule for the singular values.
        By default, it is the product of the singular values.
        Input : float list. Output : float.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Attributes
    ----------
    profile : Profile
        The profile of voters on which we run the election.
    aggregation_rule : callable
        The aggregation rule for the singular values.
        By default, it is the product of the singular values.
        Input : float list. Output : float.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.
    similarity : callable
        A similarity function that associate a pair
        of features vector to a similarity value (a float).
        Input : np.ndarray, np.ndarray.
        Output : float.

    Examples
    --------
    >>> my_profile = Profile(4, 2)
    >>> scores = [[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]]
    >>> embeddings = [[1, 0], [1, 1], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = FakeSVDRule(my_profile, np.dot)
    >>> _ = election.set_rule(np.sum)
    >>> election.ranking_
    [3, 0, 1, 2]
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
        """
        A function to update the aggregation rule used
        for the aggregation of the singular values.

        Parameters
        ----------
        aggregation_rule : callable
            The new aggregation rule for the singular values.
            Input : float list. Output : float.

        Return
        ------
        FakeSVDRule
            The object itself.
        """

        self.aggregation_rule = aggregation_rule
        self.delete_cache()
        return self

    def score_(self, candidates):
        embeddings_matrix = self.profile_.fake_covariance_matrix(candidates,
                                                                 f=self.similarity,
                                                                 square_root=self.square_root)
        s = np.linalg.eigvals(embeddings_matrix)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings_matrix)
            return matrix_rank, self.aggregation_rule(s[:matrix_rank])
        else:
            return self.aggregation_rule(s)
