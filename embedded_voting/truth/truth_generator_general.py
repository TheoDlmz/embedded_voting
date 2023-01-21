import numpy as np
from embedded_voting.truth.truth_generator import TruthGenerator


class TruthGeneratorGeneral(TruthGenerator):
    """
    A general generator for the ground truth ("true value") of each candidate.

    The true value of each candidate is independent and follow a probability distribution defined by the function `function`.

    Parameters
    ----------
    function : None -> np.ndarray float
        The function that defines the probability distribution of the true value of each candidate.
        If `None`, the normal distribution is used.

    Examples
    --------
        >>> np.random.seed(42)
        >>> truth_generator = TruthGeneratorGeneral()
        >>> truth_generator(n_candidates=3)
        array([ 0.49671415, -0.1382643 ,  0.64768854])
    """
    def __init__(self, function=None):
        if function is None:
            self.function = np.random.normal
        else:
            self.function = function

    def __call__(self, n_candidates):
        return self.function(size=n_candidates)
