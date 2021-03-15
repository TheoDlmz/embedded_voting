from embedded_voting.scoring.multiwinner.general import IterRules
import numpy as np
from embedded_voting.profile.ParametricProfile import ParametricProfile


class IterSVD(IterRules):
    """
    Iterative multiwinner rule based on a SVD aggregation rule.

    Parameters
    __________
    profile : Profile
        The profile of voters.
    k : int
        The size of the committee
    aggregation_rule : callable
        The aggregation rule for the singular values. By default, it is the maximum.
    square_root : bool
        If True, we take the square root of the score instead of the score itself for
        the embeddings matrix.
    quota : str
        The quota used for the re-weighing step. Either "droop" quota (n/(k+1) +1) or
        "classic" quota (n/k).
    take_min : bool
        If True, when the total satisfaction is less than the quota,
        we replace the quota by the total satisfaction. By default, it is set to false.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]
    >>> probability = [3/4, 1/4]
    >>> my_profile = ParametricProfile(6, 2, 100, scores, probability).set_parameters(1, 1)
    >>> election = IterSVD(my_profile, 3)
    >>> election.winners_
    [0, 3, 1]
    >>> election = IterSVD(my_profile, 4)
    >>> election.winners_
    [0, 1, 3, 2]
    >>> election.plot_weights(dim=[0, 0, 0], show=False)
    Weight / remaining candidate :  [25.0, 24.999999999999996, 24.999999999999996, 24.999999999999996]
    >>> election.features_vectors
    [array([1., 0.]), array([1., 0.]), array([0., 1.]), array([1., 0.])]
    """

    def __init__(self, profile=None, k=None, aggregation_rule=np.max,
                 square_root=True, quota="classic", take_min=False):
        self.aggregation_rule = aggregation_rule
        self.square_root = square_root
        super().__init__(profile=profile, k=k, quota=quota, take_min=take_min)

    def winner_k(self, winners):
        vectors = []
        scores = []
        if len(self.weights) == 0:
            self.weights = np.ones(self.profile_.n_voters)

        for candidate in range(self.profile_.n_candidates):
            if candidate in winners:
                scores.append(0)
                vectors.append(np.zeros(self.profile_.n_dim))
                continue
            embeddings = self.profile_.scored_embeddings(candidate, square_root=self.square_root)
            embeddings = np.dot(np.diag(self.weights), embeddings)
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

    def satisfaction(self, candidate, vec):
        return [self.profile_.scores[i, candidate] * np.dot(self.profile_.embeddings[i], vec)
                for i in range(self.profile_.n_voters)]
