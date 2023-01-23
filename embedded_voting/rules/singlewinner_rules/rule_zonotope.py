# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
from itertools import combinations
import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.utils.miscellaneous import volume_parallelepiped


class RuleZonotope(Rule):
    """
    Voting rule in which the aggregated score of
    a candidate is the volume of the zonotope described by
    his embedding matrix `M` such that `M[i] = score[i, candidate] * embeddings[i]`.
    (cf :meth:`~embedded_voting.Embeddings.times_ratings_candidate`).

    For each candidate, the rank `r` of her associated matrix is computed. The volume of the zonotope is
    the sum of the volumes of all the parallelepipeds associated to a submatrix keeping only `r` voters
    (cf. :func:`~embedded_voting.utils.miscellaneous.volume_parallelepiped`). The score of the candidate is then `(r, volume)`.

    
    Parameters
    ----------
    score_components : int
        The number of components in the aggregated
        score of every candidate. If `> 1`, we
        perform a lexical sort to obtain the ranking.
    embeddings_from_ratings: EmbeddingsFromRatings
        If no embeddings are specified in the call, this `EmbeddingsFromRatings` object is use to generate
        the embeddings from the ratings. Default: `EmbeddingsFromRatingsIdentity()`.

    Examples
    --------
    >>> ratings = Ratings([[1], [1]])
    >>> embeddings = Embeddings([[1, 0, 0], [-.5, 1, 0]], norm=False)
    >>> election = RuleZonotope()(ratings, embeddings)
    >>> election.scores_
    [(2, 1.0)]

    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
    >>> election = RuleZonotope()(ratings, embeddings)
    >>> election.scores_  # doctest: +ELLIPSIS
    [(2, 0.458...), (2, 0.424...), (2, 0.372...)]
    >>> election.ranking_
    [0, 1, 2]
    >>> election.winner_
    0
    >>> election.welfare_  # doctest: +ELLIPSIS
    [1.0, 0.605..., 0.0]
    """
    def __init__(self, embeddings_from_ratings=None):
        super().__init__(score_components=2, embeddings_from_ratings=embeddings_from_ratings)

    def _score_(self, candidate):
        m_candidate = self.embeddings_.times_ratings_candidate(self.ratings_.candidate_ratings(candidate))
        matrix_rank = np.linalg.matrix_rank(m_candidate)
        volume = np.sum([
            volume_parallelepiped(m_candidate[subset_of_voters, :])
            for subset_of_voters in combinations(range(self.embeddings_.n_voters), matrix_rank)
        ])
        return matrix_rank, volume
