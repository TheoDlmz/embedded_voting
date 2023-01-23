import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule_svd import RuleSVD


class RuleSVDMin(RuleSVD):
    """
    Voting rule in which the aggregated score of a candidate is the minimum singular value
    of his embedding matrix (cf :meth:`~embedded_voting.Embeddings.times_ratings_candidate`).

    Parameters
    ----------
    square_root: boolean
        If True, use the square root of score in the matrix.
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
    >>> election = RuleSVDMin()(ratings, embeddings)
    >>> election.scores_
    [0.5885971537535042, 0.4657304054015261, 0.5608830567730065]
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.7744377762720253]

    """
    def __init__(self, square_root=True, use_rank=False, embedded_from_ratings=None):
        super().__init__(aggregation_rule=np.min, square_root=square_root, use_rank=use_rank,
                         embedded_from_ratings=embedded_from_ratings)
