import numpy as np
from embedded_voting.truth.truth_generator import TruthGenerator


class TruthGeneratorPowerLaw(TruthGenerator):
    """
    Generate the true values of the candidates with a power law.
    """
    def __init__(self, minimum_value=10, maximum_value=20, a=5):
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.a = a

    def __call__(self, n_candidates):
        return self.minimum_value + np.random.power(self.a, size=n_candidates)*(self.maximum_value-self.minimum_value)
