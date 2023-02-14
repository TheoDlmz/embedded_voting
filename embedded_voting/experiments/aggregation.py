import numpy as np
import embedded_voting as ev
from tqdm import tqdm
from multiprocess.pool import Pool


def evaluate(list_agg, truth, testing, training, pool=None):
    """
    Run a sim.
    Parameters
    ----------
    list_agg: :class:`list`
        Rules to test.
    truth: :class:`~numpy.ndarray`
        Ground truth of testing values (n_tries X n_candidates).
    testing: :class:`~numpy.ndarray`
        Estimated scores (n_agents X n_tries X n_candidates).
    training: :class:`~numpy.ndarray`
        Training scores  (n_agents X training_size).
    pool: :class:`~mulyiprocess.pool.Pool`, optional.
        Use parallelism.
    Returns
    -------
    :class:`~numpy.ndarray`
        Efficiency of each algorithm.
    Examples
    --------
    >>> np.random.seed(42)
    >>> n_training = 10
    >>> n_tries = 100
    >>> n_c = 20
    >>> generator = make_generator()
    >>> training = generator(n_training)
    >>> testing = generator(n_tries*n_c).reshape(generator.n_voters, n_tries, n_c)
    >>> truth = generator.ground_truth_.reshape(n_tries, n_c)
    >>> list_agg = make_aggs(order=default_order+['Rand'])
    >>> with Pool() as p:
    ...     res = evaluate(list_agg=list_agg[:-1], truth=truth, testing=testing, training=training, pool=p)
    >>> ', '.join( f"{a.name}: {r:.2f}" for a, r in zip(list_agg, res) )
    'MA: 0.94, PL+: 0.89, EV+: 0.95, EV: 0.94, AV: 0.90, PV: 0.86, RV: 0.85, Single: 0.82, PL: 0.78'
    >>> res = evaluate(list_agg=list_agg, truth=truth, testing=testing, training=training)
    >>> ', '.join( f"{a.name}: {r:.2f}" for a, r in zip(list_agg, res) )
    'MA: 0.94, PL+: 0.89, EV+: 0.95, EV: 0.94, AV: 0.90, PV: 0.86, RV: 0.85, Single: 0.82, PL: 0.78, Rand: 0.49'
    """
    n_tries = testing.shape[1]
    for agg in list_agg:
        if agg.name.endswith('+'):
            _ = agg(training).winner_
            agg.train()
    results = np.zeros(len(list_agg))

    def election(bundle):
        ratings_candidates, truth_candidates = bundle
        res = np.zeros(len(list_agg))
        # Reset the trained aggregators for consistency
        for agg in list_agg:
            if agg.name[-1] != '+':
                agg.reset()
        # Welfare
        welfare = ev.RuleSumRatings()(ev.Ratings([truth_candidates])).welfare_
        # We run the aggregators, and we look at the welfare of the winner
        for k, agg in enumerate(list_agg):
            w = agg(ratings_candidates).winner_
            res[k] += welfare[w]
        return res

    if pool is not None:
        chunk_size = max(1, int(n_tries / 100))
        for result in pool.imap_unordered(election,
                                       tqdm(((testing[:, i, :], truth[i, :])
                                             for i in range(n_tries)), total=n_tries),
                                       chunksize=chunk_size):
                results += result
    else:
        for result in (election( (testing[:, i, :], truth[i, :]) )
                       for i in tqdm(range(n_tries))):
            results += result
    return results / n_tries


def make_generator(groups=None, truth=None, features=None,
                   feat_noise=1, feat_f=None,
                   dist_noise=.1, dist_f=None):
    """
    Parameters
    ----------
    groups: :class:`list` of `int`
        Sizes of each group.
    truth: :class:`TruthGenerator`, default=N(0, 1)
        Ground truth generator.
    features: :class:`~numpy.ndarray`
        Features correlations.
    feat_noise: :class:`float`, default=1.0
        Feature noise intensity.
    feat_f: `method`, default to normal law
        Feature noise distribution.
    dist_noise: :class:`float`, default=0.1
        Distinct noise intensity.
    dist_f: `method`, default to normal law
        Distinct noise distribution.
    Returns
    -------
    :class:`Generator`
        Provides grounds truth and estimates.
    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = make_generator()
    >>> ratings = generator(2)
    >>> truth = generator.ground_truth_
    >>> truth[0]
    0.4967141530112327
    >>> ratings[:, 0]
    Ratings([1.22114616, 1.09745525, 1.1986587 , 1.09806092, 1.09782972,
             1.16859892, 0.95307467, 0.97191091, 1.08817394, 1.04311958,
             1.17582742, 1.05360028, 1.00317232, 1.29096757, 1.12182506,
             1.15115551, 1.00192787, 1.08996442, 1.15549495, 1.02930333,
             2.05731381, 0.20249691, 0.23340782, 2.01575631])
    >>> truth[1]
    -0.13826430117118466
    >>> ratings[:, 1]
    Ratings([ 1.73490024,  1.51804687,  1.58119528,  1.73370001,  1.78786054,
              1.73115071,  1.70244906,  1.68390351,  1.56616168,  1.64202946,
              1.66795001,  1.81972611,  1.74837571,  1.53770987,  1.74642228,
              1.67550566,  1.64632168,  1.77518151,  1.81711384,  1.8071419 ,
             -0.23568328, -1.22689647,  0.71740695, -1.26155344])
    """
    if groups is None:
        groups = [20] + [1]*4
    if truth is None:
        truth = ev.TruthGeneratorNormal(0, 1)
    if features is None:
        features = np.eye(len(groups))
    if feat_f is None:
        feat_f = np.random.normal
    if dist_f is None:
        dist_f = np.random.normal
    generator_parameters = {
        "truth_generator": truth,
        "groups_sizes": groups,  # Number of estimators in each group
        "groups_features": features,  # Features of the groups
        "group_noise": feat_noise,  # Standard deviation of the feature noise
        "group_noise_f": feat_f,  # Distribution for feature noise
        "independent_noise": dist_noise,  # Standard deviation of the distinct noise
        "independent_noise_f": dist_f,  # Distribution for feature noise
    }
    return ev.RatingsGeneratorEpistemicGroupsMixFree(**generator_parameters)


def f_max(ratings_v, history_mean, history_std):
    """
    Parameters
    ----------
    ratings_v: :class:`~numpy.ndarray`
        Score vector.
    history_mean: :class:`float`
        Observed mean.
    history_std: :class:`float`
        Observed standard deviation
    Returns
    -------
    :class:`~numpy.ndarray`
        The positive part of the normalized scores.
    Examples
    --------
    >>> f_max(10, 5, 2)
    2.5
    >>> f_max(10, 20, 10)
    0.0
    """
    return np.maximum(0, (ratings_v - history_mean) / history_std)


def f_renorm(ratings_v, history_mean, history_std):
    """
    Parameters
    ----------
    ratings_v: :class:`~numpy.ndarray`
        Score vector.
    history_mean: :class:`float`
        Observed mean.
    history_std: :class:`float`
        Observed standard deviation
    Returns
    -------
    :class:`~numpy.ndarray`
        The scores with mean and std normalized.
    Examples
    --------
    >>> f_renorm(10, 5, 2)
    2.5
    >>> f_renorm(10, 20, 10)
    -1.0
    """
    return (ratings_v - history_mean) / history_std


class SingleEstimator:
    """
    Returns the best estimation of one given agent. Mimics a `Rule`.
    Parameters
    ----------
    i: :class:`int`
        Index of the selected agents.
    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = make_generator()
    >>> ratings = generator(7)
    >>> rule = SingleEstimator(10)
    >>> ratings[10, :]
    Ratings([ 1.2709017 ,  0.03209107,  1.98196138,  1.12347711, -1.55465272,
             -0.72448238,  0.63366952])
    >>> rule(ratings).winner_
    2
    """

    def __init__(self, i):
        self.i = i
        self.winner_ = None

    def __call__(self, ratings, embeddings=None):
        self.winner_ = np.argmax(ratings[self.i])
        return self


class RandomWinner:
    """
    Returns a random winner. Mimics a `Rule`.
    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = make_generator()
    >>> ratings = generator(7)
    >>> rule = RandomWinner()
    >>> rule(ratings).winner_
    4
    >>> rule(ratings).winner_
    3
    """

    def __init__(self):
        self.winner_ = None

    def __call__(self, ratings, embeddings=None):
        self.winner_ = np.random.randint(ratings.shape[1])
        return self


def make_aggs(groups=None, order=None, features=None, group_noise=1, distinct_noise=.1):
    """
    Crafts a list of aggregator rules.
    Parameters
    ----------
    groups: :class:`list` of `int`
        Sizes of each group (for the Model-Aware rule).
    order: :class:`list`, optional
        Short names of the aggregators to return.
    features: :class:`~numpy.ndarray`, optional
        Features correlations (for the Model-Aware rule). Default to independent groups.
    group_noise: :class:`float`, default=1.0
        Feature noise intensity.
    distinct_noise: :class:`float`, default=0.1
        Distinct noise intensity.
    Returns
    -------
    :class:`list`
        Aggregators.
    Examples
    --------
    >>> list_agg = make_aggs()
    >>> [agg.name for agg in list_agg]
    ['MA', 'PL+', 'EV+', 'EV', 'AV', 'PV', 'RV', 'Single', 'PL']
    """
    if groups is None:
        groups = [20] + [1]*4
    if order is None:
        order = default_order
    if features is None:
        features = np.eye(len(groups))
    dict_agg = {
        'MA': ev.Aggregator(rule=ev.RuleModelAware(groups,
                                                   features,
                                                   group_noise,
                                                   distinct_noise), name="MA"),
        'EV': ev.Aggregator(rule=ev.RuleFastNash(), name="EV"),
        'AV': ev.Aggregator(rule=ev.RuleRatingsHistory(rule=ev.RuleApprovalProduct(), f=f_max), name="AV"),
        'PV': ev.Aggregator(rule=ev.RuleRatingsHistory(rule=ev.RuleShiftProduct(), f=f_renorm), name="PV"),
        'RV': ev.Aggregator(rule=ev.RuleRatingsHistory(rule=ev.RuleSumRatings(), f=f_renorm), name="RV"),
        'PL': ev.Aggregator(rule=ev.RuleRatingsHistory(rule=ev.RuleMLEGaussian(), f=f_renorm), name="PL"),
        'PL+': ev.Aggregator(rule=ev.RuleRatingsHistory(rule=ev.RuleMLEGaussian(), f=f_renorm), name="PL+",
                             default_train=False, default_add=False),
        'EV+': ev.Aggregator(rule=ev.RuleFastNash(), name="EV+", default_train=False, default_add=False),
        'Single': ev.Aggregator(rule=SingleEstimator(groups[0] - 1), name="Single"),
        'Rand': ev.Aggregator(rule=RandomWinner(), name="Rand"),
    }
    return [dict_agg[k] for k in order]


default_order = ['MA', 'PL+', 'EV+', 'EV', 'AV', 'PV', 'RV', 'Single', 'PL']

colors = {"EV": "#de302a", "AV": "#32e62c", "RV": "#dee046", "PL": "#2488ed", "PL+": "#4540cf", "EV+": "#a83d3d",
          "Rand": "#686868",
          "MA": "#454545", "Single": "#808080", "PV": "#ed921a"}

default_handles = {"EV": "\\gls{ev}", "AV": "\\gls{av}", "RV": "\\gls{rv}", "PL": "\\gls{ml}", "PL+": "\\gls{ml+}",
           "EV+": "\\gls{ev+}",
           "Rand": "\\gls{rw}", "MA": "\\gls{ga}", "Single": "\\gls{sa}", "PV": "\\gls{np}"}

handles = {"EV": "EV", "AV": "AV", "RV": "RV", "PL": "PL", "PL+": "PL+",
           "EV+": "EV+",
           "Rand": "RW", "MA": "MA", "Single": "SA", "PV": "NP"}
