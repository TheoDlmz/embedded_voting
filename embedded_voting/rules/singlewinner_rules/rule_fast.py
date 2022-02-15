import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_self import EmbeddingsFromRatingsSelf
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.utils.cached import cached_property


class RuleFast(Rule):
    """
    Voting rule in which the aggregated score of
    a candidate is based on singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.

    Attributes
    ----------
    ratings_ : np.ndarray
        The ratings given by voters to candidates
    embeddings_: Embeddings
        The embeddings of the voters
    n_sing_val_: int
        The number of singular values we want to consider when computing the score
        of some candidate

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = RuleFast()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, f=None, aggregation_rule=np.prod):
        super().__init__(embeddings_from_ratings=EmbeddingsFromRatingsSelf(norm=False))
        self.aggregation_rule = aggregation_rule
        if f is None:
            self.f = lambda x: np.sqrt(np.maximum(0, x/np.linalg.norm(x)))
        else:
            self.f = f

    @cached_property
    def modified_ratings_(self):
        modified_ratings = np.array([self.f(x) for x in self.ratings_])
        return modified_ratings

    @cached_property
    def correlations_(self):
        return EmbeddingsFromRatingsCorrelation()(self.embeddings_)

    @cached_property
    def n_sing_val_(self):
        return self.correlations_.n_sing_val_

    def _score_(self, candidate):
        try:
            correlations = np.array(self.correlations_).copy()
            for i in range(self.ratings_.n_voters):
                s = self.modified_ratings_[i, candidate]
                correlations[i, :] *= s
                correlations[:, i] *= s
        except AttributeError:
            correlations = np.array(self.correlations_).copy()
            for i in range(self.ratings_.n_voters):
                s = self.modified_ratings_[i, candidate]
                correlations[i, :] *= s
            correlations = np.dot(correlations, correlations.T)

        s = np.linalg.eigvals(correlations)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        s = np.sort(s)[::-1]
        return self.aggregation_rule(s[:self.n_sing_val_])
