from embedded_voting.epistemicGenerators.ratings_generator_epistemic import RatingsGeneratorEpistemic
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
    minimum_value : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_value : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = RatingsGeneratorEpistemicMultivariate(np.ones((5, 5)))
    >>> generator()
    Ratings([[14.85728131],
             [14.85728131],
             [14.85728131],
             [14.85728131],
             [14.85728131]])
    >>> generator.independent_noise = 0.5
    >>> generator()
    Ratings([[13.81223438],
             [13.95888662],
             [13.21274843],
             [13.65293116],
             [13.98058382]])

    """

    def __init__(self, covariance_matrix, independent_noise=0, minimum_value=10, maximum_value=20):
        n_voters = len(covariance_matrix)
        super().__init__(n_voters=n_voters, minimum_value=minimum_value,
                         maximum_value=maximum_value)
        self.covariance_matrix = covariance_matrix
        self.independent_noise = independent_noise

    def __call__(self, n_candidates=1, *args):
        self.ground_truth_ = self.generate_true_values(n_candidates=n_candidates)
        ratings = np.zeros((self.n_voters, n_candidates))
        for i in range(n_candidates):
            v_dependent_noise = np.random.multivariate_normal(
                mean=np.zeros(self.n_voters), cov=self.covariance_matrix)
            v_independent_noise = np.random.normal(
                loc=0, scale=self.independent_noise, size=self.n_voters)
            ratings[:, i] = self.ground_truth_[i] + v_dependent_noise + v_independent_noise
        return Ratings(ratings)
