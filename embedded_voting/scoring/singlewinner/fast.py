import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.profile import Profile
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.profile.fastprofile import FastProfile


class Fast(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is based on singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    f : callable
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
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.
    aggregation_rule : callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.


    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> profile = FastProfile(scores)
    >>> election = Fast(profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self,  profile=None, f=None, aggregation_rule=np.prod):
        super().__init__(profile=profile)
        self.aggregation_rule = aggregation_rule
        if f is None:
            self.f = lambda x: np.sqrt(np.maximum(0, x/np.linalg.norm(x)))
        else:
            self.f = f

        self._modified_ratings = None
        if profile is not None:
            self(profile)

    def __call__(self, profile):
        self.profile_ = profile

        modified_ratings = np.zeros(profile.ratings.shape)
        for i in range(profile.n_voters):
            modified_ratings[i] = self.f(profile.ratings[i])

        self._modified_ratings = modified_ratings
        try:
            self.n_v = profile.embeddings.n_dim
        except AttributeError:

            u, s, v = np.linalg.svd(profile.embeddings.positions)
            n_voters, n_candidates = profile.embeddings.positions.shape
            s = np.sqrt(s)
            s /= s.sum()
            n_v = 0
            for s_e in s:
                if s_e >= max(1/n_voters, 1/n_candidates):
                    n_v += 1
            self.n_v = n_v

        return self

    def _score_(self, candidate):
        try:
            embeddings = self.profile_.fast_embeddings.positions.copy()
            for i in range(self.profile_.n_voters):
                s = self._modified_ratings[i, candidate]
                embeddings[i, :] *= s
                embeddings[:, i] *= s
        except AttributeError:
            embeddings = self.profile_.embeddings.positions.copy()
            for i in range(self.profile_.n_voters):
                s = self._modified_ratings[i, candidate]
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
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> profile = FastProfile(scores)
    >>> election = FastNash(profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self,  profile=None, f=None):
        super().__init__(profile=profile, f=f, aggregation_rule=np.prod)


class FastSum(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> profile = FastProfile(scores)
    >>> election = FastSum(profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, profile=None, f=None):
        super().__init__(profile=profile, f=f, aggregation_rule=np.sum)


class FastMin(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the minimum of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization f.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> profile = FastProfile(scores)
    >>> election = FastMin(profile)
    >>> election.ranking_
    [2, 0, 1]
    >>> election.winner_
    2

    """
    def __init__(self, profile=None, f=None):
        super().__init__(profile=profile, f=f, aggregation_rule=np.min)


class FastLog(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the log sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization f.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> profile = FastProfile(scores)
    >>> election = FastLog(profile)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, profile=None, f=None):
        super().__init__(profile=profile, f=f)

    def __call__(self, profile):
        super().__call__(profile)
        self.aggregation_rule = lambda x: np.sum(np.log(1+x*profile.n_voters))
        return self
