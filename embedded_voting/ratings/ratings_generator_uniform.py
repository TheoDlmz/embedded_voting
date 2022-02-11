import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.ratings.ratings_generator import RatingsGenerator


class RatingsGeneratorUniform(RatingsGenerator):
    """
    Generate uniform random ratings.

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

    def __init__(self, n_voters, minimum_rating=0, maximum_rating=1):
        super().__init__(n_voters=n_voters)
        self.minimum_rating = minimum_rating
        self.maximum_rating = maximum_rating
        self.amplitude = self.maximum_rating - self.minimum_rating

    def __call__(self, n_candidates):
        return Ratings(np.random.rand(self.n_voters, n_candidates)) * self.amplitude + self.minimum_rating
