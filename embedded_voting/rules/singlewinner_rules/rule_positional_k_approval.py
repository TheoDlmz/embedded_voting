import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.rules.singlewinner_rules.rule_positional import RulePositional
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash


class RulePositionalKApproval(RulePositional):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with k-Approval rule (vector ``[1, 1, ..., 0]``
    with `k` ones).

    Parameters
    ----------
    k : int
        The k parameter of the k-approval.
        By default, it is set to 2.
    rule : Rule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.


    Examples
    --------
    >>> ratings = np.array([[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]])
    >>> embeddings = Embeddings(np.array([[1, 0], [1, 1], [0, 1]]), norm=True)
    >>> election = RulePositionalKApproval(n_candidates=4, k=2, rule=RuleSVDNash(use_rank=True))(
    ...     ratings, embeddings)
    >>> election.fake_ratings_
    Ratings([[0., 0., 1., 1.],
             [0., 1., 1., 0.],
             [1., 1., 0., 0.]])
    >>> election.ranking_
    [1, 2, 0, 3]
    """
    def __init__(self, n_candidates, k=2, rule=None):
        if k >= n_candidates:
            raise ValueError("k should be < n_candidates")
        points = [1]*k + [0]*(n_candidates-k)
        super().__init__(points, rule)
