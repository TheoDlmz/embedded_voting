import numpy as np
from embedded_voting.rules.singlewinner_rules.rule_fast import RuleFast


class RuleFastSum(RuleFast):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = RuleFastSum()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, f=None, embeddings_as_history=False):
        super().__init__(f=f, aggregation_rule=np.sum, embeddings_as_history=embeddings_as_history)
