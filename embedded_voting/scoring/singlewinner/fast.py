import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile


class Fast(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is based on singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    function : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.


    Attributes
    ----------
    profile : Profile
        The profile of voters on which we run the election.
    function : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.
    aggregation_rule : callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.


    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = Fast(my_profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.640015624924835]

    """
    def __init__(self,  profile=None, function=None, aggregation_rule=np.prod):
        super().__init__(profile=profile)
        self.aggregation_rule = aggregation_rule
        if function is None:
            self.function = lambda x: np.sqrt(np.maximum(0, x/np.linalg.norm(x)))
        else:
            self.function = function

        if profile is not None:
            self(profile)
        self._modified_scores = None
        if profile is not None:
            self(profile)

    def __call__(self, profile):
        self.profile_ = profile

        modified_scores = np.zeros(profile.scores.shape)
        for i in range(profile.n_voters):
            modified_scores[i] = self.function(profile.scores[i])

        self._modified_scores = modified_scores
        try:
            self.n_v = profile.n_sing_val
        except AttributeError:

            u, s, v = np.linalg.svd(profile.embeddings)
            n_voters, n_candidates = profile.embeddings.shape
            s = np.sqrt(s)
            s /= s.sum()
            n_v = 0
            for s_e in s:
                if s_e >= max(1/n_voters, 1/n_candidates):
                    n_v += 1
            self.n_v = n_v

        return self

    def score_(self, candidate):
        try:
            embeddings = self.profile_.fast_embeddings.copy()
            for i in range(self.profile_.n_voters):
                s = self._modified_scores[i, candidate]
                embeddings[i, :] *= s
                embeddings[:, i] *= s
        except AttributeError:
            embeddings = self.profile_.embeddings.copy()
            for i in range(self.profile_.n_voters):
                s = self._modified_scores[i, candidate]
                embeddings[i, :] *= s
            embeddings = np.dot(embeddings, embeddings.T)

        s = np.linalg.eigvals(embeddings)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        return self.aggregation_rule(s[:self.n_v])


class FastNash(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the product of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    function : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = FastNash(my_profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.640015624924835]

    """
    def __init__(self,  profile=None, function=None):
        super().__init__(profile=profile, function=function)
        self.aggregation_rule = np.prod


class FastSum(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    function : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = FastSum(my_profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.640015624924835]

    """
    def __init__(self, profile=None, function=None):
        super().__init__(profile=profile, function=function, aggregation_rule=np.sum)


class FastMin(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the minimum of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    function : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = FastMin(my_profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.640015624924835]

    """
    def __init__(self, profile=None, function=None):
        super().__init__(profile=profile, function=function, aggregation_rule=np.min)


class FastLog(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the log sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    function : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = FastLog(my_profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.7088954658756377]

    """
    def __init__(self, profile=None, function=None):
        super().__init__(profile=profile, function=function)

    def __call__(self, profile):
        super().__call__(profile)
        self.aggregation_rule = lambda x: np.sum(np.log(1+x*profile.n_voters))
        return self
