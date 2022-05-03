import numpy as np
from embedded_voting.manipulation.individual_manipulation.manipulation_ordinal import ManipulationOrdinal
from embedded_voting.rules.singlewinner_rules.rule_instant_runoff import RuleInstantRunoff
from embedded_voting.ratings_from_embeddings.ratings_from_embeddings_correlated import RatingsFromEmbeddingsCorrelated
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash
from embedded_voting.ratings.ratings import Ratings


class ManipulationOrdinalIRV(ManipulationOrdinal):
    """
    This class do the single voter manipulation
    analysis for the :class:`RuleInstantRunoff` rule_positional.
    It is faster than the general class
    class:`ManipulationOrdinal`.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    rule : Rule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> manipulation = ManipulationOrdinalIRV(ratings, embeddings, RuleSVDNash())
    >>> manipulation.prop_manipulator_
    0.4
    >>> manipulation.avg_welfare_
    0.4
    >>> manipulation.worst_welfare_
    0.0
    >>> manipulation.manipulation_global_
    [2, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    """

    def __init__(self, ratings, embeddings, rule=None):
        ratings = Ratings(ratings)
        super().__init__(ratings, embeddings, RuleInstantRunoff(ratings.n_candidates), rule)

    def _create_fake_scores(self, eliminated, scores):
        """
        This function creates a fake ratings for each step of the IRV function

        Parameters
        ----------
        eliminated : int list
            The list of candidates already eliminated.

        scores : float list
            The scores of the candidates.

        Return
        ------
        np.ndarray
            A fake score matrix of size :attr:`n_voter`, :attr:`n_candidate`.

        """
        n_voters, n_candidates = self.ratings.shape
        fake_profile = np.zeros((n_voters, n_candidates))
        points = np.zeros(n_candidates)
        points[0] = 1

        for i in range(n_voters):
            scores_i = scores[i].copy()
            scores_i[eliminated] = 0
            ord_i = np.argsort(scores_i)[::-1]
            ord_i = np.argsort(ord_i)
            fake_profile[i] = points[ord_i]

        return fake_profile

    def manipulation_voter(self, i):
        ratings = self.ratings.copy()
        ratings_i = self.ratings[i].copy()
        preferences_order = np.argsort(ratings_i)[::-1]

        n_candidates = self.ratings.shape[1]

        if preferences_order[0] == self.winner_:
            return self.winner_

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]

        queue_eliminated = [([], -1)]

        while len(queue_eliminated) > 0:
            (el, one) = queue_eliminated.pop()

            if len(el) == n_candidates:
                winner = el[-1]
                index_candidate = np.where(preferences_order == winner)[0][0]
                if index_candidate < best_manipulation_i:
                    best_manipulation_i = index_candidate
            else:
                fake_profile = self._create_fake_scores(el, ratings)
                fake_profile[i] = np.ones(n_candidates)
                self.ratings = fake_profile
                scores_max = self.extended_rule.rule(self.ratings, self.embeddings).scores_

                fake_profile[i] = np.zeros(n_candidates)
                self.ratings = fake_profile
                scores_min = self.extended_rule.rule(self.ratings, self.embeddings).scores_

                all_scores = [(s, j, 1) for j, s in enumerate(scores_max) if j not in el]
                all_scores += [(s, j, 0) for j, s in enumerate(scores_min) if j not in el]

                all_scores.sort()

                if all_scores[0][1] == one:
                    if all_scores[1][1] == one:
                        queue_eliminated.append((el+[all_scores[0][1]], -1))
                    else:
                        queue_eliminated.append((el+[all_scores[1][1]], one))
                else:
                    queue_eliminated.append((el+[all_scores[0][1]], one))
                    if all_scores[1][2] == 0 and one == -1:
                        queue_eliminated.append((el+[all_scores[1][1]], all_scores[0][1]))

        self.ratings = ratings

        best_manipulation = preferences_order[best_manipulation_i]

        return best_manipulation
