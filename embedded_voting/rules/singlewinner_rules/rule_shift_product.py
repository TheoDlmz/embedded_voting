import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings


class RuleShiftProduct(Rule):
    """
    Voting rule in which the score of a candidate is the product of her ratings, shifted by 2, and clamped at 0.1.

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
    >>> election = RuleShiftProduct()(ratings)
    >>> election.scores_
    [14.85..., 15.60..., 14.16...]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1

    """

    def _score_(self, candidate):
        candidate_ratings = np.maximum(0.1, self.ratings_.candidate_ratings(candidate)+2)
        prod_ratings = np.prod(candidate_ratings)
        return prod_ratings
