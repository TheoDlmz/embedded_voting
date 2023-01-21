import numpy as np
from embedded_voting.ratings.ratings_generator_epistemic_groups_mix \
    import RatingsGeneratorEpistemicGroupsMix


class RatingsGeneratorEpistemicGroupsMean(RatingsGeneratorEpistemicGroupsMix):
    """
    A generator of ratings such that voters are
    separated into different groups and the noise of
    an voter on a candidate is equal to the noise
    of his group plus his own independent noise.

    This is a particular case of :class:`RatingsGeneratorEpistemicGroupsMix`
    when `groups_features` is the identity matrix, i.e. each group has its own
    exclusive feature.

    As a result, for each candidate `i`:

    * For each group, a `sigma_group` is drawn (absolute part of a normal variable, scaled by
      `group_noise`). Then a `noise_group` is drawn (normal variable scaled by `sigma_group`).
    * For each voter, `noise_dependent` is equal to the `noise_group` of her group.
    * For each voter, `noise_independent` is drawn (normal variable scaled by `independent_noise`).
    * For each voter of each group, the rating is computed as
      `ground_truth[i] + noise_dependent + noise_independent`.

    Parameters
    ----------
    groups_sizes : list or np.ndarray
        The number of voters in each groups.
        The sum is equal to :attr:`~embedded_voting.RatingsGenerator.n_voters`.
    group_noise : float
        The variance used to sample the noise of each group.
    independent_noise : float
        The variance used to sample the independent noise of each voter.
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
    >>> np.random.seed(44)
    >>> generator = RatingsGeneratorEpistemicGroupsMean([2, 2])
    >>> generator()  # doctest: +ELLIPSIS
    Ratings([[16.3490...],
             [16.3490...],
             [19.16928...],
             [19.16928...]])
    >>> generator.ground_truth_  # doctest: +ELLIPSIS
    array([17.739...])
    """
    def __init__(self, groups_sizes, group_noise=1, independent_noise=0, truth_generator=None):
        n_groups = len(groups_sizes)
        group_features = np.eye(n_groups)
        super().__init__(groups_sizes=groups_sizes, groups_features=group_features,
                         group_noise=group_noise, independent_noise=independent_noise,
                         truth_generator=truth_generator)
