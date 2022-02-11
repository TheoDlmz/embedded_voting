import itertools
import numpy as np
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting.manipulation.individual_manipulation.manipulation import Manipulation
from embedded_voting.ratings_from_embeddings.ratings_from_embeddings_correlated import RatingsFromEmbeddingsCorrelated
from embedded_voting.rules.singlewinner_rules.rule_positional_extension_borda import RulePositionalExtensionBorda
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash


class ManipulationOrdinal(Manipulation):
    """
    This class extends the :class:`Manipulation`
    class to ordinal extension (irv, borda, plurality, etc.).

    Parameters
    ----------
    ratings : Profile
        The ratings of voters on which we do the analysis.
    extension : PositionalRuleExtension
        The ordinal extension used.
    rule : Rule
        The aggregation rule we want to analysis.

    Attributes
    ----------
    rule : Rule
        The aggregation rule we want to analysis.
    winner_ : int
        The index of the winner of the election without manipulation.
    welfare_ : float list
        The welfares of the candidates without manipulation.
    extended_rule : Rule
        The rule we are analysing
    extension : PositionalRuleExtension
        The extension used.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> extension = RulePositionalExtensionBorda(3)
    >>> manipulation = ManipulationOrdinal(ratings, embeddings, extension, RuleSVDNash())
    >>> manipulation.prop_manipulator_
    0.0
    >>> manipulation.manipulation_global_
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> manipulation.avg_welfare_
    1.0
    """

    def __init__(self, ratings, embeddings, extension, rule=None):
        super().__init__(ratings, embeddings)
        self.rule = rule
        self.extension = extension
        if rule is not None:
            self.extended_rule = self.extension.set_rule(rule)
            self.extended_rule(self.ratings, self.embeddings)
            self.winner_ = self.extended_rule.winner_
            self.welfare_ = self.rule(self.ratings, self.embeddings).welfare_
            self.delete_cache()
        else:
            self.extended_rule = None

    def __call__(self, rule):
        self.rule = rule
        self.extended_rule = self.extension.set_rule(rule)
        self.extended_rule(self.ratings, self.embeddings)
        self.winner_ = self.extended_rule.winner_
        self.welfare_ = self.rule(self.ratings, self.embeddings).welfare_
        self.delete_cache()
        return self

    def manipulation_voter(self, i):
        score_i = self.ratings[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        n_candidates = self.ratings.shape[1]
        points = np.arange(n_candidates)[::-1]
        if preferences_order[0] == self.winner_:
            return self.winner_

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]

        for perm in itertools.permutations(range(n_candidates)):
            self.ratings[i] = points[list(perm)]
            fake_run = self.extended_rule(self.ratings, self.embeddings)
            new_winner = fake_run.winner_
            index_candidate = np.where(preferences_order == new_winner)[0][0]
            if index_candidate < best_manipulation_i:
                best_manipulation_i = index_candidate
                if best_manipulation_i == 0:
                    break

        best_manipulation = preferences_order[best_manipulation_i]
        self.ratings[i] = score_i

        return best_manipulation
