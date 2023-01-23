import numpy as np
from embedded_voting.manipulation.individual_manipulation.manipulation_ordinal import ManipulationOrdinal
from embedded_voting.rules.singlewinner_rules.rule_positional_k_approval import RulePositionalKApproval
from embedded_voting.ratings_from_embeddings.ratings_from_embeddings_correlated import RatingsFromEmbeddingsCorrelated
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash
from embedded_voting.ratings.ratings import Ratings


class ManipulationOrdinalKApproval(ManipulationOrdinal):
    """
    This class do the single voter manipulation
    analysis for the :class:`RulePositionalKApproval` rule_positional.
    It is faster than the general class
    class:`ManipulationOrdinal`.

    Parameters
    ----------
    ratings : Profile
        The ratings of voters on which we do the analysis.
    embeddings : Embeddings
        The embeddings of the voters.
    k : int
        The k parameter for the k-approval rule.
    rule : Rule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> manipulation = ManipulationOrdinalKApproval(ratings, embeddings, 2, RuleSVDNash())
    >>> manipulation.prop_manipulator_
    0.0
    >>> manipulation.avg_welfare_
    1.0
    >>> manipulation.worst_welfare_
    1.0
    >>> manipulation.manipulation_global_
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """

    def __init__(self, ratings, embeddings, k=2, rule=None):
        ratings = Ratings(ratings)
        super().__init__(ratings, embeddings, RulePositionalKApproval(ratings.n_candidates, k=k), rule)

    def manipulation_voter(self, i):
        fake_scores_i = self.extended_rule.fake_ratings_[i].copy()
        score_i = self.ratings[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        k = int(np.sum(self.rule_positional.points))
        n_candidates = self.ratings.n_candidates
        unk = n_candidates - k

        if preferences_order[0] == self.winner_:
            return self.winner_

        self.extended_rule.fake_ratings_[i] = np.ones(n_candidates)
        scores_max = self.extended_rule.base_rule(self.extended_rule.fake_ratings_, self.embeddings).scores_
        self.extended_rule.fake_ratings_[i] = np.zeros(n_candidates)
        scores_min = self.extended_rule.base_rule(self.extended_rule.fake_ratings_, self.embeddings).scores_
        self.extended_rule.fake_ratings_[i] = fake_scores_i

        all_scores = [(s, j, 1) for j, s in enumerate(scores_max)]
        all_scores += [(s, j, 0) for j, s in enumerate(scores_min)]

        all_scores.sort()
        all_scores = all_scores[::-1]

        best_manipulation = np.where(preferences_order == self.winner_)[0][0]

        for (_, j, kind) in all_scores:
            if kind == 0:
                break

            index_candidate = np.where(preferences_order == j)[0][0]
            if index_candidate < best_manipulation:
                k -= 1
                best_manipulation = index_candidate
            unk -= 1

            if unk < 0:
                break

        best_manipulation = preferences_order[best_manipulation]

        return best_manipulation
