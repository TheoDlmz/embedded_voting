# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.rule import Rule
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


class RuleSVD(Rule):
    """
    Voting rule in which the aggregated score of
    a candidate is based on singular values
    of his embedding matrix
    (cf :meth:`~embedded_voting.Embeddings.scored_embeddings`).

    Parameters
    ----------
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Attributes
    ----------
    aggregation_rule : callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
    >>> election = RuleSVD()(ratings, embeddings)
    >>> election.scores_  # DOCTEST: +ELLIPSIS
    [0.6041522986797..., 0.547722557505..., 0.5567764362830...]
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_  # DOCTEST: +ELLIPSIS
    [1.0, 0.0, 0.16044515869439...]

    """
    def __init__(self, aggregation_rule=np.prod, square_root=True, use_rank=False):
        score_components = 1
        if use_rank:
            score_components = 2
        super().__init__(score_components=score_components)
        self.square_root = square_root
        self.aggregation_rule = aggregation_rule
        self.use_rank = use_rank

    def _score_(self, candidate):
        embeddings = self.embeddings_.times_ratings_candidate(np.sqrt(self.ratings_.candidate_ratings(candidate)))

        if self.embeddings_.shape[1] == 0:
            return self.aggregation_rule(self.ratings_.candidate_ratings(candidate))

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

    def set_rule(self, aggregation_rule):
        """
        A function to update the aggregation rule
        :attr:`aggregation_rule`
        used for the singular values.

        Parameters
        ----------
        aggregation_rule : callable
            The new aggregation rule for the singular values.
            Input : float list. Output : float.

        Return
        ------
        RuleSVD
            The object itself.

        Examples
        --------
        >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
        >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
        >>> election = RuleSVD()(ratings, embeddings)
        >>> election.ranking_
        [0, 2, 1]
        >>> election.set_rule(np.sum)
        <embedded_voting.scoring.singlewinner.rule_svd.RuleSVD object at ...>
        >>> election.ranking_
        [1, 0, 2]
        """
        self.aggregation_rule = aggregation_rule
        self.delete_cache()
        return self
