# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""

from embedded_voting.rules.singlewinner_rules.rule import Rule


class RuleCut(Rule):
    """
    This rule perform another rule but on a subset of voters
    """

    def __init__(self, rule, cut, score_components=1, embeddings_from_ratings=None):
        super().__init__(score_components, embeddings_from_ratings)
        self.cut = cut
        self.rule = rule

    def __call__(self, ratings, embeddings=None):
        cut = self.cut
        return self.rule(ratings[:cut], embeddings[:cut])

    def _score_(self, candidate):
        return self.rule.score_(candidate)

