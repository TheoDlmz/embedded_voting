from embedded_voting.ratings.ratings_generator_epistemic \
    import RatingsGeneratorEpistemic
import numpy as np
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicMultivariate(RatingsGeneratorEpistemic):
    """
    A generator of ratings based on a covariance matrix.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix of the voters.
        Should be of shape :attr:`~embedded_voting.RatingsGenerator.n_voters`,
        :attr:`~embedded_voting.RatingsGenerator.n_voters`.
    independent_noise : float
        The variance of the independent noise.
    truth_generator : TruthGenerator
        The truth generator used to generate to true values of each candidate.
        Default: `TruthGeneratorUniform(10, 20)`.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = RatingsGeneratorEpistemicMultivariate(np.ones((5, 5)))
    >>> generator()  # doctest: +ELLIPSIS
    Ratings([[17.2428...],
             [17.2428...],
             [17.2428...],
             [17.2428...],
             [17.2428...]])
    >>> generator.independent_noise = 0.5
    >>> generator()  # doctest: +ELLIPSIS
    Ratings([[14.5710...],
             [14.3457...],
             [15.0093...],
             [14.3981...],
             [14.1460...]])
    """

    def __init__(self, covariance_matrix, independent_noise=0, truth_generator=None):
        n_voters = len(covariance_matrix)
        super().__init__(n_voters=n_voters, truth_generator=truth_generator)
        self.covariance_matrix = covariance_matrix
        self.independent_noise = independent_noise

    def __call__(self, n_candidates=1):
        self.ground_truth_ = self.truth_generator(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            v_dependent_noise = np.random.multivariate_normal(
                mean=np.zeros(self.n_voters), cov=self.covariance_matrix)
            v_independent_noise = np.random.normal(
                loc=0, scale=self.independent_noise, size=self.n_voters)
            ratings[:, i] = self.ground_truth_[i] + v_dependent_noise + v_independent_noise
        return Ratings(ratings)
