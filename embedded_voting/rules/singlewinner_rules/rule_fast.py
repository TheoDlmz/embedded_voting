import numpy as np
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.embeddings.embeddings_correlation import EmbeddingsCorrelation
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import center_and_normalize


class RuleFast(Rule):
    """
    Voting rule in which the aggregated score of a candidate is based on singular values
    of his score matrix.

    Parameters
    ----------
    embeddings_from_ratings: EmbeddingsFromRatingsCorrelation
        If no embeddings are specified in the call, this `EmbeddingsFromRatings` object is use to generate
        the embeddings from the ratings. Default:
        `EmbeddingsFromRatingsCorrelation(preprocess_ratings=center_and_normalize)`.
    f : callable
        The transformation for the ratings given by each voter.
        Input : (ratings_v: np.ndarray, history_mean: Number, history_std: Number).
        Output : modified_ratings_v: np.ndarray.
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : list of float. Output : float.
        By default, it is the product of the singular values.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = RuleFast()(ratings)
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    """
    def __init__(self, embeddings_from_ratings=None, f=None, aggregation_rule=np.prod):
        if embeddings_from_ratings is None:
            embeddings_from_ratings = EmbeddingsFromRatingsCorrelation(preprocess_ratings=center_and_normalize)
        if f is None:
            def f(ratings_v, history_mean, history_std):
                if ratings_v.sum() == 0:
                    return ratings_v
                else:
                    return np.sqrt(np.maximum(0, 2 + (ratings_v - history_mean) / history_std))
        super().__init__(embeddings_from_ratings=embeddings_from_ratings)
        self.f = f
        self.aggregation_rule = aggregation_rule

    def __call__(self, ratings, embeddings: EmbeddingsCorrelation = None):
        return super().__call__(ratings=ratings, embeddings=embeddings)

    @cached_property
    def modified_ratings_(self):
        """Ratings: Modified ratings. For each voter, `f` is applied to her original ratings."""
        ratings = self.ratings_
        history_means = self.embeddings_.ratings_means
        history_stds = self.embeddings_.ratings_stds
        return Ratings([
            self.f(ratings_v, history_mean, history_std)
            for ratings_v, history_mean, history_std in zip(ratings, history_means, history_stds)
        ])

    def _score_(self, candidate):
        m_c_dot_m_c_t = np.array(self.embeddings_) * np.outer(self.modified_ratings_[:, candidate],
                                                              self.modified_ratings_[:, candidate])

        s = np.linalg.eigvals(m_c_dot_m_c_t)
        s = np.maximum(s, 0)
        s = np.sqrt(s)
        s = np.sort(s)[::-1]
        return self.aggregation_rule(s[:self.embeddings_.n_sing_val])
