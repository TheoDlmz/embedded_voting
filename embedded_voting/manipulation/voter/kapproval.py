import numpy as np
from embedded_voting.manipulation.voter.general import SingleVoterManipulationExtension
from embedded_voting.scoring.singlewinner.ordinal import KApprovalExtension
from embedded_voting.profile.generator import CorrelatedRatings
from embedded_voting.embeddings.generator import ParametrizedEmbeddings
from embedded_voting.scoring.singlewinner.svd import SVDNash


class SingleVoterManipulationKApp(SingleVoterManipulationExtension):
    """
    This class do the single voter manipulation
    analysis for the :class:`KApprovalExtension` extension.
    It is faster than the general class
    class:`SingleVoterManipulationExtension`.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    k : int
        The k parameter for the k-approval rule.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = ParametrizedEmbeddings(10, 3)(.8)
    >>> ratings = CorrelatedRatings(3, 3, scores_matrix)(embeddings, .8)
    >>> manipulation = SingleVoterManipulationKApp(ratings, embeddings, 2, SVDNash())
    >>> manipulation.prop_manipulator_
    0.2
    >>> manipulation.avg_welfare_
    0.8
    >>> manipulation.worst_welfare_
    0.0
    >>> manipulation.manipulation_global_
    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1]
    """

    def __init__(self, ratings, embeddings, k=2, rule=None):
        if not isinstance(ratings, np.ndarray):
            ratings = ratings.ratings
        super().__init__(ratings, embeddings, KApprovalExtension(ratings.shape[1], k=k), rule)

    def manipulation_voter(self, i):
        fake_scores_i = self.extended_rule.fake_ratings[i].copy()
        score_i = self.ratings[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        k = int(np.sum(self.extension.points))
        n_candidates = self.ratings.shape[1]
        unk = n_candidates - k

        if preferences_order[0] == self.winner_:
            return self.winner_

        self.extended_rule.fake_ratings[i] = np.ones(n_candidates)
        scores_max = self.extended_rule.base_rule(self.extended_rule.fake_ratings, self.embeddings).scores_
        self.extended_rule.fake_ratings[i] = np.zeros(n_candidates)
        scores_min = self.extended_rule.base_rule(self.extended_rule.fake_ratings, self.embeddings).scores_
        self.extended_rule.fake_ratings[i] = fake_scores_i

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
