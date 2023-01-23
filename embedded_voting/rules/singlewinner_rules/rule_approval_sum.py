import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings


class RuleApprovalSum(Rule):
    """
    Voting rule in which the score of a candidate is the number of approval (vote greater than 0)
    that it gets. Ties are broken by sum of score (range voting).

    More precisely, her score is a tuple whose components are:

    * The number of her nonzero ratings.
    * The sum of her ratings.

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
    >>> election = RuleApprovalSum()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.scores_
    [(3, 1.4), (2, 1.6), (3, 1.3)]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.9285714285714287]
    """

    def __init__(self, embeddings_from_ratings=None):
        super().__init__(score_components=2, embeddings_from_ratings=embeddings_from_ratings)

    def _score_(self, candidate):
        candidate_ratings = self.ratings_.candidate_ratings(candidate)
        mask = candidate_ratings > 0
        n_nonzero_ratings = np.sum(mask)
        sum_ratings = np.sum(candidate_ratings[mask])
        return n_nonzero_ratings, sum_ratings
