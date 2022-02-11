import numpy as np
from embedded_voting.ratings.ratings_generator_epistemic \
    import RatingsGeneratorEpistemic
from embedded_voting.ratings.ratings import Ratings


class RatingsGeneratorEpistemicLinearGaussian(RatingsGeneratorEpistemic):
    """Generator based on Gaussian noises, multiplied by a matrix.

    For each candidate `c`, a vector of "elementary noises" is drawn i.i.d. following a normal
    Gaussian distribution. Then the ratings for candidate `c` are computed as
    `ratings[:, c] = m_voters_noises @ noises_c`.

    Parameters
    ----------
    m_voters_noises: list or np.ndarray
        An array of size `n_voters` * `n_noises`, where `n_noises` is the number of elementary
        gaussian noises.
    truth_generator : TruthGenerator
        The truth generator used to generate to true values of each candidate.
        Default: `TruthGeneratorUniform(10, 20)`.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.

    Examples
    --------
    >>> np.random.seed(42)
    >>> n_voters = 5
    >>> n_noises = 3
    >>> m_voters_noises = np.random.randn(n_voters, n_noises)
    >>> ratings_generator = RatingsGeneratorEpistemicLinearGaussian(m_voters_noises)
    >>> ratings_generator(n_candidates=2)  # doctest: +ELLIPSIS
    Ratings([[15.888..., 10.7850...],
             [15.645...,  9.6529...],
             [14.252..., 11.0540...],
             [16.573..., 10.1916...],
             [19.074...,  8.254... ]])
    >>> ratings_generator.ground_truth_  # doctest: +ELLIPSIS
    array([16.118..., 11.394...])
    """

    def __init__(self, m_voters_noises, truth_generator=None):
        self.m_voters_noises = np.array(m_voters_noises)
        self.n_voters, self.n_noises = m_voters_noises.shape
        super().__init__(n_voters=self.n_voters, truth_generator=truth_generator)

    def __call__(self, n_candidates=1):
        self.ground_truth_ = self.truth_generator(n_candidates=n_candidates)
        m_noises_candidates = np.random.randn(self.n_noises, n_candidates)
        return Ratings(
            self.ground_truth_[np.newaxis, :]
            + self.m_voters_noises @ m_noises_candidates
        )
