import numpy as np
from embedded_voting.truth.truth_generator import TruthGenerator


class TruthGeneratorGeneral(TruthGenerator):
    """
    Generate the true values of the candidates with a normal law.
    """
    def __init__(self, function=None):
        if function is None:
            self.function = np.random.normal()
        else:
            self.function = function

    def __call__(self, n_candidates):
        return self.function(size=n_candidates)
