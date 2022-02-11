# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""

import numpy as np
from embedded_voting.scoring.singlewinner.rule import Rule
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings


class RuleSumRatings(Rule):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the scores given by
    the voters.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
    >>> election = RuleSumRatings()(ratings, embeddings)
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
