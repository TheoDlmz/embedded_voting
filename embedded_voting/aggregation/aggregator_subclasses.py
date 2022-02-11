from embedded_voting.aggregation.aggregator import Aggregator
from embedded_voting.scoring.singlewinner.rule_fast_nash import RuleFastNash
from embedded_voting.scoring.singlewinner.rule_fast_sum import RuleFastSum
from embedded_voting.scoring.singlewinner.rule_sum_ratings import RuleSumRatings
from embedded_voting.scoring.singlewinner.rule_product_ratings import RuleProductRatings
from embedded_voting.scoring.singlewinner.rule_mle_gaussian import RuleMLEGaussian
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_self import EmbeddingsFromRatingsSelf


class AggregatorFastNash(Aggregator):
    def __init__(self):
        super().__init__(RuleFastNash(),
                         default_train=True, name="RuleFastNash")


class AggregatorFastSum(Aggregator):
    def __init__(self):
        super().__init__(RuleFastSum(),
                         default_train=True, name="RuleFastSum")


class AggregatorSumRatings(Aggregator):
    def __init__(self):
        super().__init__(RuleSumRatings(), embedder=EmbeddingsFromRatingsSelf(),
                         default_train=False, name="RuleSumRatings")


class AggregatorProduct(Aggregator):
    def __init__(self):
        super().__init__(RuleProductRatings(), embedder=EmbeddingsFromRatingsSelf(),
                         default_train=False, name="RuleProductRatings")


class AggregatorMLEGaussian(Aggregator):
    def __init__(self):
        super().__init__(RuleMLEGaussian(), embedder=EmbeddingsFromRatingsSelf(),
                         default_train=True, name="RuleMLEGaussian")
