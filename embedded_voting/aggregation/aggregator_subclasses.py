from embedded_voting.aggregation.aggregator import Aggregator
from embedded_voting.rules.singlewinner_rules.rule_fast_nash import RuleFastNash
from embedded_voting.rules.singlewinner_rules.rule_fast_sum import RuleFastSum
from embedded_voting.rules.singlewinner_rules.rule_sum_ratings import RuleSumRatings
from embedded_voting.rules.singlewinner_rules.rule_approval_product import RuleApprovalProduct
from embedded_voting.rules.singlewinner_rules.rule_mle_gaussian import RuleMLEGaussian
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_covariance import EmbeddingsFromRatingsCovariance


class AggregatorFastNash(Aggregator):
    def __init__(self, default_train=True):
        super().__init__(rule=RuleFastNash(), default_train=default_train, name="RuleFastNash")


class AggregatorFastSum(Aggregator):
    def __init__(self, default_train=True):
        super().__init__(rule=RuleFastSum(), default_train=default_train, name="RuleFastSum")


class AggregatorSumRatings(Aggregator):
    def __init__(self):
        super().__init__(rule=RuleSumRatings(), default_train=False, name="RuleSumRatings")


class AggregatorProductRatings(Aggregator):
    def __init__(self):
        super().__init__(rule=RuleApprovalProduct(), default_train=False, name="RuleProductRatings")


class AggregatorMLEGaussian(Aggregator):
    def __init__(self, default_train=True):
        super().__init__(rule=RuleMLEGaussian(), embeddings_from_ratings=EmbeddingsFromRatingsCovariance(),
                         default_train=default_train, name="RuleMLEGaussian")
