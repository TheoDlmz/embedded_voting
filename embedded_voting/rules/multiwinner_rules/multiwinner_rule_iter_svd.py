import numpy as np
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting.ratings_from_embeddings.ratings_from_embeddings_correlated import RatingsFromEmbeddingsCorrelated
from embedded_voting.rules.multiwinner_rules.multiwinner_rule_iter import MultiwinnerRuleIter


class MultiwinnerRuleIterSVD(MultiwinnerRuleIter):
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
        the :meth:`~embedded_voting.Embeddings.scored_embeddings`.
    quota : str
        The quota used for the re-weighing step.
        Either ``'droop'`` quota `(n/(k+1) +1)` or
        ``'classic'`` quota `(n/k)`.
    take_min : bool
        If True, when the total
        satisfaction is less than the :attr:`~embedded_voting.MultiwinnerRuleIter.quota`,
        we replace the quota by the total satisfaction.
        By default, it is set to False.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = np.array([[1, 0.8, 0.5, 0, 0, 0], [0, 0, 0, 0.5, 0.8, 1]])
    >>> probability = [3/4, 1/4]
    >>> embeddings = EmbeddingsGeneratorPolarized(100, 2, probability)(1)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=1, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> election = MultiwinnerRuleIterSVD(3)(ratings, embeddings)
    >>> election.winners_
    [0, 1, 5]
    >>> _ = election.set_k(4)
    >>> election.winners_
    [0, 1, 5, 2]
    >>> election.plot_weights(dim=[0, 0, 0], show=False)
    Weight / remaining candidate :  [25.0, 24.99999999999999, 24.999999999999996, 30.999999999999993]
    >>> election.features_vectors
    Embeddings([[1., 0.],
                [1., 0.],
                [0., 1.],
                [1., 0.]])
    """

    def __init__(self, k=None, aggregation_rule=np.max, square_root=True, quota="classic", take_min=False):
        self.aggregation_rule = aggregation_rule
        self.square_root = square_root
        super().__init__(k=k, quota=quota, take_min=take_min)

    def _winner_k(self, winners):
        vectors = []
        scores = []

        n_candidates = self.ratings.n_candidates
        n_dim = self.embeddings.n_dim
        for candidate in range(n_candidates):
            if candidate in winners:
                scores.append(0)
                vectors.append(np.zeros(n_dim))
                continue
            if self.square_root:
                embeddings = self.embeddings.times_ratings_candidate(np.sqrt(self.ratings.candidate_ratings(candidate)))
            else:
                embeddings = self.embeddings.times_ratings_candidate(self.ratings.candidate_ratings(candidate))

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
