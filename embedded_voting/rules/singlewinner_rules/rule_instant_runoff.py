import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_identity import EmbeddingsFromRatingsIdentity
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash


class RuleInstantRunoff(Rule):
    """
    This class enables to extend a
    voting rule to an ordinal input
    with Instant Runoff ranking. You cannot access
    to the :attr:`~embedded_voting.ScoringRule.scores_` because IRV only
    compute the ranking of the candidates.

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
    >>> election = RuleInstantRunoff(RuleSVDNash())(ratings, embeddings)
    >>> election.ranking_
    [1, 0, 2, 3]
    """

    def __init__(self,  rule=None):
        super().__init__()
        self.rule = rule

    def __call__(self, ratings, embeddings=None):
        ratings = Ratings(ratings)
        if embeddings is None:
            embeddings = EmbeddingsFromRatingsIdentity()(ratings)
        self.embeddings_ = Embeddings(embeddings, norm=True)
        self.ratings_ = ratings
        self.delete_cache()
        return self

    def set_rule(self, rule):
        self.rule = rule
        self.delete_cache()
        return self

    def _score_(self, candidate):
        raise NotImplementedError

    @cached_property
    def ranking_(self):
        n_candidates = self.ratings_.n_candidates
        ranking = np.zeros(n_candidates, dtype=int)
        eliminated = []
        for i in range(n_candidates):
            fake_ratings = self._create_fake_ratings(eliminated)
            rule_i = self.rule(fake_ratings, self.embeddings_)
            loser = rule_i.ranking_[n_candidates-1-i]
            ranking[n_candidates-i-1] = loser
            eliminated.append(loser)
        return list(ranking)

    @cached_property
    def winner_(self):
        return self.ranking_[0]

    def _create_fake_ratings(self, eliminated):
        """
        This function creates a fake ratings for the election, based
        on the candidates already eliminated during the previous
        steps.

        Return
        ------
        np.ndarray
            The fake ratings.
        """
        fake_ratings = np.zeros(self.ratings_.shape)
        points = np.zeros(self.ratings_.n_candidates)
        points[0] = 1

        for i in range(self.ratings_.n_voters):
            scores_i = self.ratings_[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_ratings[i] = points[ord_i]

        return fake_ratings
