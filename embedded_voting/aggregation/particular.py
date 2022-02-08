from embedded_voting.aggregation.aggregator import Aggregator
from embedded_voting.scoring.singlewinner.fast import FastNash, FastSum
from embedded_voting.scoring.singlewinner.trivialRules import SumScores, ProductScores
from embedded_voting.scoring.singlewinner.mlerules import MLEGaussian
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_self import EmbeddingsFromRatingsSelf


class AggregatorFastNash(Aggregator):
    def __init__(self):
        super().__init__(FastNash(), default_train=True, name="FastNash")


class AggregatorFastSum(Aggregator):
    def __init__(self):
        super().__init__(FastSum(), default_train=True, name="FastSum")


class AggregatorSum(Aggregator):
    def __init__(self):
        super().__init__(SumScores(), embedder=EmbeddingsFromRatingsSelf(), default_train=False, name="SumScores")


class AggregatorProduct(Aggregator):
    def __init__(self):
        super().__init__(ProductScores(), embedder=EmbeddingsFromRatingsSelf(),
                         default_train=False, name="ProductScores")


class AggregatorMLEGaussian(Aggregator):
    def __init__(self):
        super().__init__(MLEGaussian(), embedder=EmbeddingsFromRatingsSelf(), default_train=True, name="MLEGaussian")
