import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule


class RuleApprovalSum(Rule):
    """
    Voting rule in which the score of a candidate is the number of approval (vote greater than 0)
    that it gets. Ties are broken by sum of score (range voting)

    More precisely, her score is a tuple whose components are:

    * The number of her nonzero ratings.
    * The sum of her ratings.

    No embeddings are used for this rule.
    """

    def __init__(self, embeddings_from_ratings=None):
        super().__init__(score_components=2, embeddings_from_ratings=embeddings_from_ratings)

    def _score_(self, candidate):
        candidate_ratings = self.ratings_.candidate_ratings(candidate)
        mask = candidate_ratings > 0
        n_nonzero_ratings = np.sum(mask)
        sum_ratings = np.sum(candidate_ratings[mask])
        return n_nonzero_ratings, sum_ratings
