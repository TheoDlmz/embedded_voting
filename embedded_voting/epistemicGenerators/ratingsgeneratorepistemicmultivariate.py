from embedded_voting.epistemicGenerators.ratingsgeneratorepistemic import RatingsGeneratorEpistemic
import numpy as np
from embedded_voting.ratings.ratings import Ratings

class RatingsGeneratorEpistemicMultivariate(RatingsGeneratorEpistemic):
    """
    A generator of scores based on a covariance matrix.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix of the voters.
        Should be of shape :attr:`~embedded_voting.ScoreGenerator.n_voters`,
        :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    independent_noise : float
        The variance of the independent noise.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix of the voters.
        Should be of shape :attr:`~embedded_voting.ScoreGenerator.n_voters`,
        :attr:`~embedded_voting.ScoreGenerator.n_voters`.
    independent_noise : float
        The variance of the independent noise.

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = RatingsGeneratorEpistemicMultivariate(np.ones((5, 5)))
    >>> generator()
    (array([13.74540119]), Ratings([[14.85728131],
             [14.85728131],
             [14.85728131],
             [14.85728131],
             [14.85728131]]))
    >>> generator.independent_noise = 0.5
    >>> generator()
    (array([12.9122914]), Ratings([[13.81223...],
             [13.95888...],
             [13.21274...],
             [13.65293...],
             [13.98058...]]))

    """
    def __init__(self, covariance_matrix, independent_noise=0, minimum_score=10, maximum_score=20):
        n_voters = len(covariance_matrix)
        super().__init__(n_voters, minimum_score, maximum_score)
        self.covariance_matrix = covariance_matrix
        self.independent_noise = independent_noise

    def __call__(self, n_candidates=1, *args):
        scores = np.zeros((self.n_voters, n_candidates))
        truth = np.zeros(n_candidates)
        for i in range(n_candidates):
            truth_i = self.generate_true_score()
            truth[i] = truth_i
            scores_i = np.random.multivariate_normal(np.ones(self.n_voters)*truth_i, self.covariance_matrix)
            scores_i += np.random.randn(self.n_voters) * self.independent_noise
            scores[:, i] = scores_i

        return truth, Ratings(scores)
