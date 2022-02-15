import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


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
    n_v: int
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
        super().__init__()
        self.aggregation_rule = aggregation_rule
        if f is None:
            self.f = lambda x: np.sqrt(np.maximum(0, x/np.linalg.norm(x)))
        else:
            self.f = f

        self._modified_ratings = None

    def __call__(self, ratings, embeddings=None):
        ratings = Ratings(ratings)
        modified_ratings = np.zeros(ratings.shape)
        for i in range(ratings.n_voters):
            modified_ratings[i] = self.f(ratings.voter_ratings(i))
        self.ratings_ = ratings
        self._modified_ratings = modified_ratings

        embeddings_from_ratings = EmbeddingsFromRatingsCorrelation()
        if embeddings is None:
            self.embeddings_ = self.ratings_
        else:
            self.embeddings_ = np.concatenate([embeddings, self.ratings_], axis=1)
        self.correlations_ = embeddings_from_ratings(self.embeddings_)

        self.n_v = self.correlations_.n_sing_val_  # embeddings_from_ratings.n_sing_val_
        self.delete_cache()

        return self

    def _score_(self, candidate):
        try:
            correlations = np.array(self.correlations_).copy()
            for i in range(self.ratings_.n_voters):
                s = self._modified_ratings[i, candidate]
                correlations[i, :] *= s
                correlations[:, i] *= s
        except AttributeError:
            correlations = np.array(self.correlations_).copy()
            for i in range(self.ratings_.n_voters):
                s = self._modified_ratings[i, candidate]
                correlations[i, :] *= s
            correlations = np.dot(correlations, correlations.T)

        s = np.linalg.eigvals(correlations)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        s = np.sort(s)[::-1]
        return self.aggregation_rule(s[:self.n_v])
