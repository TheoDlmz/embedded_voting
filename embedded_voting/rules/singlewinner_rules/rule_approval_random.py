import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings


class RuleApprovalRandom(Rule):
    """
    Voting rule in which the score of a candidate is the number of approval (vote greater than 0)
    that it gets. Ties are broken at random.

    More precisely, her score is a tuple whose components are:

    * The number of her nonzero ratings.
    * A random value.

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
    >>> np.random.seed(42)
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> election = RuleApprovalRandom()(ratings)
    >>> election.ranking_
    [2, 0, 1]
    >>> election.scores_
    [(3, 0.3745401188473625), (2, 0.9507143064099162), (3, 0.7319939418114051)]
    >>> election.winner_
    2
    >>> election.welfare_
    [0.5116710637256354, 0.0, 1.0]

    """

    def __init__(self, embeddings_from_ratings=None):
        super().__init__(score_components=2, embeddings_from_ratings=embeddings_from_ratings)

    def _score_(self, candidate):
        candidate_ratings = self.ratings_.candidate_ratings(candidate)
        mask = candidate_ratings > 0
        n_nonzero_ratings = np.sum(mask)
        random_val = np.random.rand()
        return n_nonzero_ratings, random_val
