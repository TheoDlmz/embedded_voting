import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings

from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.embeddings.embeddings_correlation import EmbeddingsCorrelation
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import center_and_normalize


class RuleRatingsHistory(Rule):
    """
    Rule that use the ratings history to improve the embeddings, in particular the quality of the mean
    and deviation of ratings for every voter. The original rule is then applied to the modified ratings.

    Parameters
    ----------
    rule : Rule
        The rule to apply to the modified ratings.
    embeddings_from_ratings : EmbeddingsFromRatings
        The function to convert ratings to embeddings.
    f : callable
        The function to apply to the ratings. It takes as input the ratings, the mean and the standard deviation of
        the ratings in the historic. It returns the modified ratings.
        By default, it is set to `f(ratings_v, history_mean, history_std) = np.sqrt(np.maximum(0, (ratings_v - history_mean) / history_std))`.

    """
    def __init__(self, rule, embeddings_from_ratings=None, f=None):
        if embeddings_from_ratings is None:
            embeddings_from_ratings = EmbeddingsFromRatingsCorrelation(preprocess_ratings=center_and_normalize)
        if f is None:
            def f(ratings_v, history_mean, history_std):
                return np.sqrt(np.maximum(0, (ratings_v - history_mean) / history_std))
        super().__init__(embeddings_from_ratings=embeddings_from_ratings)
        self.f = f
        self.rule = rule

    def __call__(self, ratings, embeddings: EmbeddingsCorrelation = None):
        res = super().__call__(ratings=ratings, embeddings=embeddings)
        self.rule(self.modified_ratings_, self.embeddings_)
        return res

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
        return self.rule.score_(candidate)
