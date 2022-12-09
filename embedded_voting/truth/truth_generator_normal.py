import numpy as np
from embedded_voting.truth.truth_generator import TruthGenerator


class TruthGeneratorNormal(TruthGenerator):
    """
    Generate the true values of the candidates with a normal law.
    """
    def __init__(self, center=15, noise=5):
        self.center = center
        self.noise = noise
        self.rng = np.random.default_rng()

    def __call__(self, n_candidates):
        return self.rng.normal(self.center, self.noise, size=n_candidates)
