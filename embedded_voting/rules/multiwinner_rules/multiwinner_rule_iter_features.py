import numpy as np
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting.ratings_from_embeddings.ratings_from_embeddings_correlated import RatingsFromEmbeddingsCorrelated
from embedded_voting.rules.multiwinner_rules.multiwinner_rule_iter import MultiwinnerRuleIter


class MultiwinnerRuleIterFeatures(MultiwinnerRuleIter):
    """
    Iterative multiwinner rule
    based on the :class:`RuleFeatures`
    aggregation rule.

    Parameters
    ----------
    k : int
        The size of the committee.
    quota : str
        The quota used for the re-weighing step.
        Either ``'droop'`` quota `(n/(k+1) +1)` or
        ``'classic'`` quota `(n/k)`.
    take_min : bool
        If True, when the total
        satisfaction is less than the :attr:`quota`,
        we replace the quota by the total
        satisfaction. By default, it is set to False.
    
    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = np.array([[1, 0.8, 0.5, 0, 0, 0], [0, 0, 0, 0.5, 0.8, 1]])
    >>> probability = [3/4, 1/4]
    >>> embeddings = EmbeddingsGeneratorPolarized(100, 2, probability)(1)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=1, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> election = MultiwinnerRuleIterFeatures(3)(ratings, embeddings)
    >>> election.winners_
    [0, 5, 1]
    >>> _ = election.set_k(4)
    >>> election.winners_
    [0, 5, 1, 2]
    >>> election.plot_weights(dim=[0, 0, 0], show=False)
    Weight / remaining candidate :  [25.0, 24.999999999999986, 27.999999999999993, 30.999999999999986]
    >>> election.features_vectors
    Embeddings([[1., 0.],
                [0., 1.],
                [1., 0.],
                [1., 0.]])
    """

    @staticmethod
    def compute_features(embeddings, scores):
        """
        A function to compute features
        for some embeddings and scores.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings of the voters.
            Should be of shape :attr:`~embedded_voting.Embeddings.n_voters`,
            :attr:`~embedded_voting.Embeddings.embeddings.n_dim`.
        scores : np.ndarray
            The scores given by the voters to the candidates.
            Should be of shape :attr:`~embedded_voting.Ratings.n_voters`,
            :attr:`~embedded_voting.Ratings.n_candidates`.

        Return
        ------
        np.ndarray
            The features of every candidates.
            Of shape :attr:`~embedded_voting.Ratings.n_candidates`,
            :attr:`~embedded_voting.Embeddings.n_dim`.
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(embeddings.T, embeddings)), embeddings.T), scores).T

    def _winner_k(self, winners):

        features = self.compute_features(self.embeddings, np.dot(np.diag(self.weights), self.ratings))
        scores = np.sum(features ** 2, axis=1)

        scores = np.array(scores)
        scores[winners] = 0

        winner_j = np.argmax(scores)
        vec = features[winner_j]

        return winner_j, vec
