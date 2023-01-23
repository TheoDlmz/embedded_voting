# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings


class RuleSumRatings(Rule):
    """
    Voting rule in which the score of a candidate is the sum of her ratings.

    No embeddings are used for this rule.
    
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
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> election = RuleSumRatings()(ratings)
    >>> election.scores_
    [1.4, 1.6, 1.3]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.3333333333333328, 1.0, 0.0]
    """
    def _score_(self, candidate):
        return self.ratings_.candidate_ratings(candidate).sum()
