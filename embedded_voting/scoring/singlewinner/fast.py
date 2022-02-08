import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_correlation import EmbeddingsFromRatingsCorrelation
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


class Fast(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is based on singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    embeddings_as_history: bool
        If true, the embeddings are considered as rating history and used to
        compute the real embeddings


    Attributes
    ----------
    ratings_ : np.ndarray
        The ratings given by voters to candidates
    embeddings_: Embeddings
        The embeddings of the voters
    n_v: int
        The number of singular values we want to consider when computing the score
        of some candidate
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.
    aggregation_rule : callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    embeddings_as_history: bool
        If true, the embeddings are considered as rating history and used to
        compute the real embeddings


    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = Fast()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self,  f=None, aggregation_rule=np.prod, embeddings_as_history=False):
        super().__init__()
        self.aggregation_rule = aggregation_rule
        if f is None:
            self.f = lambda x: np.sqrt(np.maximum(0, x/np.linalg.norm(x)))
        else:
            self.f = f

        self._modified_ratings = None
        self.embeddings_as_history = embeddings_as_history

    def __call__(self, ratings, embeddings=None):
        ratings = Ratings(ratings)
        modified_ratings = np.zeros(ratings.shape)
        for i in range(ratings.n_voters):
            modified_ratings[i] = self.f(ratings.voter_ratings(i))
        self.ratings_ = ratings
        self._modified_ratings = modified_ratings

        if self.embeddings_as_history or embeddings is None:
            embedder = EmbeddingsFromRatingsCorrelation()
            if embeddings is None:
                self.embeddings_ = embedder(self.ratings_)
            else:
                self.embeddings_ = embedder(np.concatenate([embeddings, self.ratings_], axis=1))
        else:
            self.embeddings_ = Embeddings(embeddings)
            self.embeddings_.n_sing_val_ = embeddings.n_sing_val_

        self.n_v = self.embeddings_.n_sing_val_ #embedder.n_sing_val_
        self.delete_cache()

        return self

    def _score_(self, candidate):
        try:
            embeddings = np.array(self.embeddings_).copy()
            for i in range(self.ratings_.n_voters):
                s = self._modified_ratings[i, candidate]
                embeddings[i, :] *= s
                embeddings[:, i] *= s
        except AttributeError:
            embeddings = np.array(self.embeddings_).copy()
            for i in range(self.ratings_.n_voters):
                s = self._modified_ratings[i, candidate]
                embeddings[i, :] *= s
            embeddings = np.dot(embeddings, embeddings.T)

        s = np.linalg.eigvals(embeddings)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        s = np.sort(s)[::-1]
        return self.aggregation_rule(s[:self.n_v])


class FastNash(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the product of the important singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = FastNash()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self,  f=None, embeddings_as_history=False):
        super().__init__(f=f, aggregation_rule=np.prod, embeddings_as_history=embeddings_as_history)


class FastSum(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization function.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = FastSum()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, f=None, embeddings_as_history=False):
        super().__init__(f=f, aggregation_rule=np.sum, embeddings_as_history=embeddings_as_history)


class FastMin(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the minimum of the important singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization f.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = FastMin()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self,  f=None, embeddings_as_history=False):
        super().__init__(f=f, aggregation_rule=np.min, embeddings_as_history=embeddings_as_history)


class FastLog(Fast):
    """
    Voting rule in which the aggregated score of
    a candidate is the log sum of the important singular values
    of his score matrix.

    Parameters
    ----------
    f : callable
        The transformation for the scores given by the voters.
        Input : np.ndarray. Output : np.ndarray
        By default, it is the normalization f.

    Examples
    --------
    >>> ratings = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> election = FastLog()(ratings)
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0

    """
    def __init__(self, f=None, embeddings_as_history=False):
        super().__init__(f=f, embeddings_as_history=embeddings_as_history)

    def __call__(self, ratings, embeddings=None):
        ratings = Ratings(ratings)
        self.aggregation_rule = lambda x: np.sum(np.log(1+x*ratings.n_voters))
        super().__call__(ratings, embeddings)
        return self
