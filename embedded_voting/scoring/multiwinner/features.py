
from embedded_voting.scoring.multiwinner.general import IterRule
from embedded_voting.embeddings.generator import EmbeddingsGeneratorPolarized
from embedded_voting.ratings.ratingsFromEmbeddings import RatingsFromEmbeddingsCorrelated
import numpy as np


class IterFeatures(IterRule):
    """
    Iterative multiwinner rule
    based on the :class:`FeaturesRule`
    aggregation rule.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    >>> probability = [3/4, 1/4]
    >>> embeddings = EmbeddingsGeneratorPolarized(100, 2, probability)(1)
    >>> ratings = RatingsFromEmbeddingsCorrelated(6, 2, scores_matrix)(embeddings, 1)
    >>> election = IterFeatures(3)(ratings, embeddings)
    >>> election.winners_
    [0, 3, 1]
    >>> _ = election.set_k(4)
    >>> election.winners_
    [0, 3, 1, 2]
    >>> election.plot_weights(dim=[0, 0, 0], show=False)
    Weight / remaining candidate :  [25.0, 24.999999999999996, 25.000000000000004, 24.999999999999996]
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
