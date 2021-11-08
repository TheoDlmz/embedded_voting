from embedded_voting.scoring.multiwinner.general import IterRule
import numpy as np
from embedded_voting.profile.generator import CorrelatedRatings
from embedded_voting.embeddings.generator import ParametrizedEmbeddings


class IterSVD(IterRule):
    """
    Iterative multiwinner rule based on
    a SVD aggregation rule.

    Parameters
    __________
    k : int
        The size of the committee.
    aggregation_rule : callable
        The aggregation rule for the singular values.
        By default, it is the maximum.
    square_root : bool
        If True, we take the square root of
        the scores instead of the scores for
        the :meth:`~embedded_voting.Profile.scored_embeddings`.
    quota : str
        The quota used for the re-weighing step.
        Either ``'droop'`` quota `(n/(k+1) +1)` or
        ``'classic'`` quota `(n/k)`.
    take_min : bool
        If True, when the total
        satisfaction is less than the :attr:`~embedded_voting.IterRule.quota`,
        we replace the quota by the total satisfaction.
        By default, it is set to False.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    >>> probability = [3/4, 1/4]
    >>> embeddings = ParametrizedEmbeddings(100, 2, probability)(1)
    >>> ratings = CorrelatedRatings(6, 2, scores_matrix)(embeddings, 1)
    >>> election = IterSVD(3)(ratings, embeddings)
    >>> election.winners_
    [0, 1, 3]
    >>> _ = election.set_k(4)
    >>> election.winners_
    [0, 1, 3, 2]
    >>> election.plot_weights(dim=[0, 0, 0], show=False)
    Weight / remaining candidate :  [25.0, 24.999999999999996, 24.999999999999996, 24.999999999999996]
    >>> election.features_vectors
    [array([1., 0.]), array([1., 0.]), array([0., 1.]), array([1., 0.])]
    """

    def __init__(self, k=None, aggregation_rule=np.max, square_root=True, quota="classic", take_min=False):
        self.aggregation_rule = aggregation_rule
        self.square_root = square_root
        super().__init__(k=k, quota=quota, take_min=take_min)

    def _winner_k(self, winners):
        vectors = []
        scores = []

        n_candidates = self.ratings.shape[1]
        n_dim = self.embeddings.positions.shape[1]
        for candidate in range(n_candidates):
            if candidate in winners:
                scores.append(0)
                vectors.append(np.zeros(n_dim))
                continue
            if self.square_root:
                embeddings = self.embeddings.scored(np.sqrt(self.ratings[::, candidate]))
            else:
                embeddings = self.embeddings.scored(self.ratings[::, candidate])

            weights = self.weights

            if self.square_root:
                weights = np.sqrt(weights)
            embeddings = np.dot(np.diag(weights), embeddings)

            _, s_values, s_vectors = np.linalg.svd(embeddings, full_matrices=False)
            scores.append(self.aggregation_rule(s_values))
            vec = s_vectors[0]
            if vec.sum() < 0:
                vec = -vec
            vectors.append(vec)

        scores = np.array(scores)
        scores[winners] = 0

        winner_j = np.argmax(scores)
        vec = vectors[winner_j]

        return winner_j, vec
