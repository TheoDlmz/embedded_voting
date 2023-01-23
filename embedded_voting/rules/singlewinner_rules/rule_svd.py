# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.utils.miscellaneous import singular_values_short


class RuleSVD(Rule):
    """
    Voting rule in which the aggregated score of a candidate is based on singular values
    of his embedding matrix (cf :meth:`~embedded_voting.Embeddings.times_ratings_candidate`).

    Implicitly, ratings are assumed to be nonnegative.

    Parameters
    ----------
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    square_root: boolean
        If True, use the square root of ratings in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.
    embedded_from_ratings: EmbeddingsFromRatings
        If no embeddings are specified in the call, this `EmbeddingsFromRatings` object is use to generate
        the embeddings from the ratings. Default: `EmbeddingsFromRatingsIdentity()`.

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
    def __init__(self, aggregation_rule=np.prod, square_root=True, use_rank=False, embedded_from_ratings=None):
        self.aggregation_rule = aggregation_rule
        self.square_root = square_root
        self.use_rank = use_rank
        score_components = 2 if use_rank else 1
        super().__init__(score_components=score_components, embeddings_from_ratings=embedded_from_ratings)

    def _score_(self, candidate):
        if self.square_root:
            m_candidate = self.embeddings_.times_ratings_candidate(np.sqrt(self.ratings_.candidate_ratings(candidate)))
        else:
            m_candidate = self.embeddings_.times_ratings_candidate(self.ratings_.candidate_ratings(candidate))
        s = singular_values_short(m_candidate)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(m_candidate)
            return matrix_rank, self.aggregation_rule(s[:matrix_rank])
        else:
            return self.aggregation_rule(s)
