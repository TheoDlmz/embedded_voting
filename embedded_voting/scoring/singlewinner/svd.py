# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile


class SVDRule(ScoringRule):
    """
    Voting rule based on singular values of the embedding matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voter on which we run the election
    aggregation_rule: callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Attributes
    ----------
    profile : Profile
        The profile of voter on which we run the election
    aggregation_rule : callable
        The aggregation rule for the singular values.
        Input : float list. Output : float.
        By default, it is the product of the singular values.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDRule(my_profile)
    >>> election.scores_
    [0.806225774829855, 0.547722557505166, 0.5567764362830023]
    >>> election.ranking_
    array([0, 2, 1], dtype=int64)
    >>> election.winner_
    0
    >>> election.welfare_
    array([1.        , 0.        , 0.03502424])

    """
    def __init__(self, profile=None, aggregation_rule=np.prod, square_root=True, use_rank=False):
        super().__init__(profile=profile)
        self.square_root = square_root
        self.aggregation_rule = aggregation_rule
        self.use_rank = use_rank
        if use_rank:
            self._score_components = 2
        else:
            self._score_components = 1

    def score_(self, candidate):
        embeddings = self.profile_.scored_embeddings(candidate, square_root=self.square_root)

        if embeddings.shape[0] < embeddings.shape[1]:
            embeddings_matrix = embeddings.dot(embeddings.T)
        else:
            embeddings_matrix = embeddings.T.dot(embeddings)

        s = np.linalg.eigvals(embeddings_matrix)
        s = np.maximum(s, np.zeros(len(s)))
        s = np.sqrt(s)
        if self.use_rank:
            matrix_rank = np.linalg.matrix_rank(embeddings)
            return matrix_rank, self.aggregation_rule(s[:matrix_rank])
        else:
            return self.aggregation_rule(s)

    def set_rule(self, aggregation_rule):
        """
        A function to update the aggregation rule used for the singular values.

        Parameters
        ----------
        aggregation_rule : callable
            The new aggregation rule for the singular values.
            Input : float list. Output : float.

        Examples
        --------
        >>> my_profile = Profile(3, 2)
        >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
        >>> embeddings = [[1, 1], [1, 0], [0, 1]]
        >>> _ = my_profile.add_voters(embeddings, scores)
        >>> election = SVDRule(my_profile)
        >>> election.ranking_
        array([0, 2, 1], dtype=int64)
        >>> election.set_rule(np.sum)
        <embedded_voting.scoring.singlewinner.svd.SVDRule object at ...>
        >>> election.ranking_
        array([0, 1, 2], dtype=int64)
        """
        self.aggregation_rule = aggregation_rule
        self.delete_cache()
        return self


class SVDNash(SVDRule):
    """
    Voting rule based on the product of the singular values of the embedding matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voter on which we run the election
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDNash(my_profile)
    >>> election.scores_
    [0.806225774829855, 0.547722557505166, 0.5567764362830023]
    >>> election.ranking_
    array([0, 2, 1], dtype=int64)
    >>> election.winner_
    0
    >>> election.welfare_
    array([1.        , 0.        , 0.03502424])

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.prod, square_root=square_root, use_rank=use_rank)


class SVDSum(SVDRule):
    """
    Voting rule based on the sum of the singular values of the embedding matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voter on which we run the election
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDSum(my_profile)
    >>> election.scores_
    [1.8200141619393269, 1.6417810801109665, 1.5535613514007114]
    >>> election.ranking_
    array([0, 1, 2], dtype=int64)
    >>> election.winner_
    0
    >>> election.welfare_
    array([1.       , 0.3310895, 0.       ])

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.sum, square_root=square_root, use_rank=use_rank)


class SVDMin(SVDRule):
    """
    Voting rule based on the minimum of the singular values of the embedding matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voter on which we run the election
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDMin(my_profile)
    >>> election.scores_
    [0.7620641440477796, 0.4657304054015261, 0.5608830567730065]
    >>> election.ranking_
    array([0, 2, 1], dtype=int64)
    >>> election.winner_
    0
    >>> election.welfare_
    array([1.        , 0.        , 0.32109962])

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.min, square_root=square_root, use_rank=use_rank)


class SVDMax(SVDRule):
    """
    Voting rule based on the maximum of the singular values of the embedding matrix.

    Parameters
    ----------
    profile: Profile
        The profile of voter on which we run the election
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDMax(my_profile)
    >>> election.scores_
    [1.0579500178915473, 1.1760506747094404, 0.9926782946277048]
    >>> election.ranking_
    array([1, 0, 2], dtype=int64)
    >>> election.winner_
    1
    >>> election.welfare_
    array([0.35595177, 1.        , 0.        ])

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.max, square_root=square_root, use_rank=use_rank)


class SVDLog(SVDRule):
    """
    Voting rule based on the sum of `log(1 + sigma/C)` where sigma are the singular values of the embedding matrix
    and C is a constant.

    Parameters
    ----------
    profile: Profile
        The profile of voter on which we run the election
    const : float
        The constant by which we divide the singular values in the log.
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.

    Examples
    --------
    >>> my_profile = Profile(3, 2)
    >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
    >>> embeddings = [[1, 1], [1, 0], [0, 1]]
    >>> _ = my_profile.add_voters(embeddings, scores)
    >>> election = SVDLog(my_profile)
    >>> election.scores_
    [1.2881962813428856, 1.1598653051965206, 1.1347313336962574]
    >>> election.ranking_
    array([0, 1, 2], dtype=int64)
    >>> election.winner_
    0
    >>> election.welfare_
    array([1.        , 0.16377663, 0.        ])

    """
    def __init__(self, profile=None, const=1, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=lambda x: np.sum(np.log(1+x/const)),
                         square_root=square_root, use_rank=use_rank)
