import numpy as np
from embedded_voting.rules.singlewinner_rules.rule_fast import RuleFast


class RuleFastNash(RuleFast):
    """
    Voting rule in which the aggregated score of
    a candidate is the product of the important singular values
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

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = RuleFastNash()(ratings)
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    """
    def __init__(self, embeddings_from_ratings=None, f=None):
        super().__init__(embeddings_from_ratings=embeddings_from_ratings, f=f, aggregation_rule=np.prod)
