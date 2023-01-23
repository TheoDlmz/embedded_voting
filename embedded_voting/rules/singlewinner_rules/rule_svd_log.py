import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule_svd import RuleSVD


class RuleSVDLog(RuleSVD):
    """
    Voting rule in which the aggregated score of a candidate is the sum of `log(1 + sigma/const)`
    where sigma are the singular values of his embedding matrix and `const` is a constant.

    Parameters
    ----------
    const : float
        The constant by which we divide
        the singular values in the log.
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
    >>> election = RuleSVDLog()(ratings, embeddings)
    >>> election.scores_
    [1.169125718695728, 1.1598653051965206, 1.1347313336962574]
    >>> election.ranking_
    [0, 1, 2]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.7307579856610341, 0.0]

    """
    def __init__(self, const=1, square_root=True, use_rank=False, embedded_from_ratings=None):
        self.const = const
        super().__init__(
            aggregation_rule=lambda x: np.sum(np.log(1 + x / self.const)),
            square_root=square_root, use_rank=use_rank, embedded_from_ratings=embedded_from_ratings
        )
