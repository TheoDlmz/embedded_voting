import numpy as np
from embedded_voting.truth.truth_generator import TruthGenerator


class TruthGeneratorNormal(TruthGenerator):
    """
    A normal generator for the ground truth ("true value") of each candidate.

    The true value of each candidate is independent and follow a Gaussian distribution with mean `center` and standard deviation `noise`.

    Parameters
    ----------
    center : float
        The mean of the Gaussian distribution.
    noise : float
        The standard deviation of the Gaussian distribution.

    Examples
    --------
        >>> np.random.seed(42)
        >>> truth_generator = TruthGeneratorNormal(center=15, noise=5)
        >>> truth_generator(n_candidates=3)
        array([17.48357077, 14.30867849, 18.23844269])

    """
    def __init__(self, center=15, noise=5):
        self.center = center
        self.noise = noise

    def __call__(self, n_candidates):
        return np.random.normal(self.center, self.noise, size=n_candidates)
