import numpy as np
from embedded_voting.epistemicGenerators.ratingsgeneratorepistemic import RatingsGeneratorEpistemic
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicLinearGaussian(RatingsGeneratorEpistemic):
    """Generator based on Gaussian noises, multiplied by a matrix.

    For each candidate `c`, a vector of "elementary noises" is drawn i.i.d. following a normal
    Gaussian distribution. Then the ratings for candidate `c` are computed as
    `ratings[:, c] = array_voters_noises @ noises_c`.

    Parameters
    ----------
    array_voters_noises: list or np.ndarray
        An array of size `n_voters` * `n_noises`, where `n_noises` is the number of elementary
        gaussian noises.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Examples
    --------
    >>> np.random.seed(42)
    >>> n_voters = 5
    >>> n_noises = 3
    >>> array_voters_noises = np.random.randn(n_voters, n_noises)
    >>> ratings_generator = RatingsGeneratorEpistemicLinearGaussian(array_voters_noises)
    >>> truth, ratings = ratings_generator(n_candidates=2)
    >>> truth
    array([16.11852895, 11.39493861])
    >>> ratings
    Ratings([[15.88827124, 10.78500054],
             [15.64570651,  9.65299338],
             [14.25270256, 11.05406235],
             [16.57309146, 10.19162356],
             [19.07405492,  8.2545536 ]])
    """

    def __init__(self, array_voters_noises, minimum_score=10, maximum_score=20):
        self.array_voters_noises = np.array(array_voters_noises)
        self.n_voters, self.n_noises = array_voters_noises.shape
        super().__init__(
            n_voters=self.n_voters,
            minimum_score=minimum_score,
            maximum_score=maximum_score
        )

    def __call__(self, n_candidates=1, *args):
        self.ground_truth_ = np.array([self.generate_true_score() for _ in range(n_candidates)])
        array_noises_candidates = np.random.randn(self.n_noises, n_candidates)
        ratings = Ratings(
            self.ground_truth_[np.newaxis, :]
            + self.array_voters_noises @ array_noises_candidates
        )
        return ratings
