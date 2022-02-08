import numpy as np
from embedded_voting import RatingsGenerator, Ratings


class RatingsGeneratorUniform(RatingsGenerator):
    """
    A ratings generator that generates random ratings between 0 and 1 for a given number of voters
    and candidates.

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = RatingsGeneratorUniform(n_voters=5)
    >>> generator(n_candidates=4)
    Ratings([[0.37454012, 0.95071431, 0.73199394, 0.59865848],
             [0.15601864, 0.15599452, 0.05808361, 0.86617615],
             [0.60111501, 0.70807258, 0.02058449, 0.96990985],
             [0.83244264, 0.21233911, 0.18182497, 0.18340451],
             [0.30424224, 0.52475643, 0.43194502, 0.29122914]])
    """

    def __call__(self, n_candidates, **kwargs):
        return Ratings(np.random.rand(self.n_voters, n_candidates))
