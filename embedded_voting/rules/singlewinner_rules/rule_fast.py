import numpy as np
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_self import EmbeddingsFromRatingsSelf
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property


class RuleFast(Rule):
    """
    Voting rule in which the aggregated score of a candidate is based on singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the ratings given by each voter.
        Input : np.ndarray. Output : np.ndarray.
        By default, we normalize and take the non-negative part.
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : list of float. Output : float.
        By default, it is the product of the singular values.

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
            f = lambda x: np.sqrt(np.maximum(0, x/np.linalg.norm(x)))
        self.f = f

    @cached_property
    def modified_ratings_(self):
        """Ratings: Modified ratings. For each voter, `f` is applied to her original ratings."""
        return Ratings([self.f(x) for x in self.ratings_])

    @cached_property
    def correlations_(self):
        """Embeddings: `EmbeddingsFromRatingsCorrelation` applied to `self.embeddings_`."""
        return EmbeddingsFromRatingsCorrelation()(self.embeddings_)

    @cached_property
    def n_sing_val_(self):
        """int: The number of singular values we want to consider when computing the score
        of some candidate."""
        return self.correlations_.n_sing_val_

    def _score_(self, candidate):
        correlations = np.array(self.correlations_).copy()
        for i in range(self.ratings_.n_voters):
            s = self.modified_ratings_[i, candidate]
            correlations[i, :] *= s
            correlations[:, i] *= s
        s = np.linalg.eigvals(correlations)
        s = np.maximum(s, 0)
        s = np.sqrt(s)
        s = np.sort(s)[::-1]
        return self.aggregation_rule(s[:self.n_sing_val_])
