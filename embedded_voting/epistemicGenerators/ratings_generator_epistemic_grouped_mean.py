import numpy as np
from embedded_voting.epistemicGenerators.ratings_generator_epistemic_grouped_mix \
    import RatingsGeneratorEpistemicGroupedMix


class RatingsGeneratorEpistemicGroupedMean(RatingsGeneratorEpistemicGroupedMix):
    """
    A generator of ratings such that voters are
    separated into different groups and the noise of
    an voter on an alternative is equal to the noise
    of his group plus his own independent noise.

    This is a particular case of :class:`RatingsGeneratorEpistemicGroupedMix`
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
    minimum_value : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_value : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.

    Examples
    --------
    >>> np.random.seed(44)
    >>> generator = RatingsGeneratorEpistemicGroupedMean([2, 2])
    >>> generator()
    Ratings([[17.625...],
             [17.625...],
             [17.422...],
             [17.422...]])
    >>> generator.ground_truth_
    array([18.348...])
    """
    def __init__(self, groups_sizes, group_noise=1, independent_noise=0, minimum_value=10,
                 maximum_value=20):
        n_groups = len(groups_sizes)
        group_features = np.eye(n_groups)
        super().__init__(groups_sizes=groups_sizes, groups_features=group_features,
                         group_noise=group_noise, independent_noise=independent_noise,
                         minimum_value=minimum_value, maximum_value=maximum_value)
