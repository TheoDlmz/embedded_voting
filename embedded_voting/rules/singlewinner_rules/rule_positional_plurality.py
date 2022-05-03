import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.rules.singlewinner_rules.rule_positional import RulePositional
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash


class RulePositionalPlurality(RulePositional):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with Plurality rule (vector ``[1, 0, ..., 0]``).

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
    >>> election = RulePositionalPlurality(4, rule=RuleSVDNash(use_rank=True))(ratings, embeddings)
    >>> election.fake_ratings_
    Ratings([[0., 0., 0., 1.],
             [0., 1., 0., 0.],
             [1., 0., 0., 0.]])
    >>> election.ranking_
    [0, 1, 3, 2]
    """

    def __init__(self, n_candidates,  rule=None):
        points = [1] + [0]*(n_candidates-1)
        super().__init__(points, rule)
