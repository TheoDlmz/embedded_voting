import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings


class RuleShiftProduct(Rule):
    """
    """

    def __init__(self, embeddings_from_ratings=None):
        super().__init__(score_components=1, embeddings_from_ratings=embeddings_from_ratings)

    def _score_(self, candidate):
        candidate_ratings = np.maximum(0.1, self.ratings_.candidate_ratings(candidate)+2)
        prod_ratings = np.prod(candidate_ratings)
        return prod_ratings
