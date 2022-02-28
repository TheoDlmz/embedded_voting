import numpy as np
from embedded_voting.rules.singlewinner_rules.rule_fast import RuleFast
from embedded_voting.ratings.ratings import Ratings


class RuleFastLog(RuleFast):
    """
    Voting rule in which the aggregated score of
    a candidate is the log sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization f.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = RuleFastLog()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, f=None):
        super().__init__(f=f)

    def __call__(self, ratings, embeddings=None):
        ratings = Ratings(ratings)
        self.aggregation_rule = lambda x: np.sum(np.log(1+x*ratings.n_voters))
        super().__call__(ratings, embeddings)
        return self
