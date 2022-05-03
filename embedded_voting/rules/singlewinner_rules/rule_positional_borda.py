import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.rules.singlewinner_rules.rule_positional import RulePositional
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash


class RulePositionalBorda(RulePositional):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with Borda rule (vector ``[m-1, m-2, ..., 1, 0]``).

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
    >>> election = RulePositionalBorda(n_candidates=4, rule=RuleSVDNash())(ratings, embeddings)
    >>> election.fake_ratings_
    Ratings([[0.        , 0.33333333, 0.66666667, 1.        ],
             [0.33333333, 1.        , 0.66666667, 0.        ],
             [1.        , 0.66666667, 0.        , 0.33333333]])
    >>> election.ranking_
    [1, 3, 2, 0]
    """
    def __init__(self, n_candidates, rule=None):
        points = [n_candidates-i-1 for i in range(n_candidates)]
        super().__init__(points, rule)
