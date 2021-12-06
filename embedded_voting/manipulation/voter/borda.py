import numpy as np
from embedded_voting.manipulation.voter.general import SingleVoterManipulationExtension
from embedded_voting.scoring.singlewinner.ordinal import BordaExtension
from embedded_voting.ratings.ratingsFromEmbeddings import RatingsFromEmbeddingsCorrelated
from embedded_voting.embeddings.generator import EmbeddingsGeneratorPolarized
from embedded_voting.scoring.singlewinner.svd import SVDNash
from embedded_voting.ratings.ratings import Ratings

class SingleVoterManipulationBorda(SingleVoterManipulationExtension):
    """
    This class do the single voter manipulation
    analysis for the :class:`BordaExtension` extension.
    It is faster than the general class
    class:`SingleVoterManipulationExtension`.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
    >>> manipulation = SingleVoterManipulationBorda(ratings, embeddings, SVDNash())
    >>> manipulation.prop_manipulator_
    0.3
    >>> manipulation.avg_welfare_
    0.7
    >>> manipulation.worst_welfare_
    0.0
    >>> manipulation.manipulation_global_
    [1, 2, 1, 1, 2, 2, 1, 1, 1, 1]
    """
    def __init__(self, ratings, embeddings, rule=None):
        ratings = Ratings(ratings)
        super().__init__(ratings, embeddings, BordaExtension(ratings.n_candidates, rule), rule)

    def manipulation_voter(self, i):
        fake_scores_i = self.extended_rule.fake_ratings_[i].copy()
        score_i = self.ratings.voter_ratings(i).copy()
        preferences_order = np.argsort(score_i)[::-1]

        n_candidates = self.ratings.n_candidates

        if preferences_order[0] == self.winner_:
            return self.winner_

        all_scores = []
        for e in range(n_candidates):
            self.extended_rule.fake_ratings_[i] = np.ones(n_candidates) * (e / (n_candidates - 1))
            altered_scores = self.extended_rule.base_rule(self.extended_rule.fake_ratings_, self.embeddings).scores_
            all_scores += [(s, j, e) for j, s in enumerate(altered_scores)]

        self.extended_rule.fake_ratings_[i] = fake_scores_i
        all_scores.sort()
        all_scores = all_scores[::-1]

        buckets = np.arange(n_candidates)

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]
        for (_, j, kind) in all_scores:
            buckets[kind] -= 1
            if buckets[kind] < 0:
                break

            if kind == (n_candidates-1):
                index_candidate = np.where(preferences_order == j)[0][0]
                if index_candidate < best_manipulation_i:
                    best_manipulation_i = index_candidate

        best_manipulation = preferences_order[best_manipulation_i]

        return best_manipulation
