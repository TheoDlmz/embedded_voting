import numpy as np
from embedded_voting.truth.truth_generator import TruthGenerator


class TruthGeneratorUniform(TruthGenerator):
    """
    A uniform generator for the ground truth ("true value") of each candidate.

    The true value of each candidate is independent and uniform in [`minimum_value`, `maximum_value`].

    Parameters
    ----------
    minimum_value : Number
        The minimum true value of a candidate.
    maximum_value : Number
        The maximum true value of a candidate.

    Examples
    --------
        >>> np.random.seed(42)
        >>> truth_generator = TruthGeneratorUniform(minimum_value=10, maximum_value=20)
        >>> truth_generator(n_candidates=3)
        array([13.74540119, 19.50714306, 17.31993942])
    """

    def __init__(self, minimum_value=10, maximum_value=20):
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value

    def __call__(self, n_candidates):
        return (
            self.minimum_value
            + np.random.rand(n_candidates) * (self.maximum_value - self.minimum_value)
        )
