import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.rules.singlewinner_rules.rule_positional import RulePositional
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash


class RulePositionalVeto(RulePositional):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with Veto rule (vector ``[1, ..., 1, 0]``).

    Parameters
    ----------
    rule : Rule
        The aggregation rule used to
        determine the aggregated scores
        of the candidates.

    Examples
    --------
    >>> ratings = np.array([[.1, .2, .8, 1], [.7, .9, .8, .6], [1, .6, .1, .3]])
    >>> embeddings = Embeddings(np.array([[1, 0], [1, 1], [0, 1]]), norm=True)
    >>> election = RulePositionalVeto(n_candidates=4, rule=RuleSVDNash())(ratings, embeddings)
    >>> election.fake_ratings_
    Ratings([[0., 1., 1., 1.],
             [1., 1., 1., 0.],
             [1., 1., 0., 1.]])
    >>> election.ranking_
    [1, 3, 0, 2]
    """
    def __init__(self, n_candidates, rule=None):
        points = [1]*(n_candidates-1) + [0]
        super().__init__(points, rule)
