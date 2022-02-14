import numpy as np
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


class RuleMaxCube(Rule):
    """
    Voting rule in which the aggregated score of
    a candidate is the volume of a cube
    described by :attr:`~embedded_voting.Embeddings.embeddings.n_dim` rows of
    the candidate embedding matrix `M` such
    that `M[i] = score[i, candidate] * embeddings[i]`.
    (cf :meth:`~embedded_voting.Embeddings.scored_embeddings`).


    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
    >>> election = RuleMaxCube()(ratings, embeddings)
    >>> election.scores_
    [(2, 0.24...), (2, 0.42...), (2, 0.16...)]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_  # doctest: +ELLIPSIS
    [0.305..., 1.0, 0.0]

    >>> ratings = Ratings([[1, 10], [1, 10], [1, 0]])
    >>> embeddings = Embeddings([[1, 0, 0], [0, 1, 0], [0, 0, 1]], norm=False)
    >>> election = RuleMaxCube()(ratings, embeddings)
    >>> election.scores_  # doctest: +ELLIPSIS
    [(3, 1.0), (2, 100.0...)]
    >>> election.scores_focus_on_last_
    [1.0, 0]
    """
    def __init__(self):
        super().__init__(score_components=2)

    def _score_(self, candidate):
        n_voters, n_dim = self.embeddings_.shape
        embeddings = self.embeddings_.times_ratings_candidate(self.ratings_.candidate_ratings(candidate))
        matrix_rank = np.linalg.matrix_rank(embeddings)
        volume = 0
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n_voters - matrix_rank:
            current_embeddings = embeddings[current_subset, ...]
            volume = max(volume, np.sqrt(np.linalg.det(np.dot(current_embeddings, current_embeddings.T))))
            x = 1
            while current_subset[matrix_rank - x] == n_voters - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return matrix_rank, volume
