# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.Profile import Profile
from embedded_voting.utils.cached import cached_property
import matplotlib.pyplot as plt
from embedded_voting.utils.miscellaneous import normalize


class SVDRule(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is based on singular values
    of his embedding matrix
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
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
        The profile of voters on which we run the election.
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
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.03502424020689993]

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
        A function to update the aggregation rule
        :attr:`aggregation_rule`
        used for the singular values.

        Parameters
        ----------
        aggregation_rule : callable
            The new aggregation rule for the singular values.
            Input : float list. Output : float.

        Return
        ------
        SVDRule
            The object itself.

        Examples
        --------
        >>> my_profile = Profile(3, 2)
        >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
        >>> embeddings = [[1, 1], [1, 0], [0, 1]]
        >>> _ = my_profile.add_voters(embeddings, scores)
        >>> election = SVDRule(my_profile)
        >>> election.ranking_
        [0, 2, 1]
        >>> election.set_rule(np.sum)
        <embedded_voting.scoring.singlewinner.svd.SVDRule object at ...>
        >>> election.ranking_
        [0, 1, 2]
        """
        self.aggregation_rule = aggregation_rule
        self.delete_cache()
        return self


class SVDNash(SVDRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the product of the singular values
    of his embedding matrix
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
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
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.03502424020689993]

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.prod, square_root=square_root, use_rank=use_rank)


class SVDSum(SVDRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of the singular values
    of his embedding matrix
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
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
    [0, 1, 2]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.3310895033605581, 0.0]

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.sum, square_root=square_root, use_rank=use_rank)


class SVDMin(SVDRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the minimum singular value
    of his embedding matrix
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
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
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.32109962168387557]

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.min, square_root=square_root, use_rank=use_rank)


class SVDMax(SVDRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the maximum singular value
    of his embedding matrix
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
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
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.3559517700252816, 1.0, 0.0]

    """
    def __init__(self, profile=None, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=np.max, square_root=square_root, use_rank=use_rank)

    def _feature(self, candidate):
        """
        A function to get the feature vector
        of the candidate passed as parameter.
        The feature vector is defined as the
        singular vector associated to the
        maximal singular value.

        Parameters
        ----------
        candidate : int
            The index of the candidate for
            which we want the feature vector.

        Return
        ------
        np.ndarray
            The feature vector of the
            candidate, of length :attr:`~embedded_voting.Profile.n_dim`.
        """
        embeddings = self.profile_.scored_embeddings(candidate, square_root=self.square_root)
        _, vp, vec = np.linalg.svd(embeddings)
        vec = vec[0]
        if vec.sum() < 0:
            return - vec * np.sqrt(vp[0])
        else:
            return vec * np.sqrt(vp[0])

    @cached_property
    def features_(self):
        """
        A function to get the feature vectors
        of all the candidates. The feature vector is
        defined as the singular vector associated
        to the maximal singular value.

        Return
        ------
        np.ndarray
            The feature vectors of all the candidates,
            of shape :attr:`~embedded_voting.Profile.n_candidates`, :attr:`~embedded_voting.Profile.n_dim`.

        Examples
        --------
        >>> my_profile = Profile(3, 2)
        >>> scores = [[.5, .6, .3], [.7, 0, .2], [.5, 1, .8]]
        >>> embeddings = [[1, 1], [1, 0], [0, 1]]
        >>> _ = my_profile.add_voters(embeddings, scores)
        >>> election = SVDMax(my_profile)
        >>> election.features_
        array([[0.8517226 , 0.57664428],
               [0.28947845, 1.04510904],
               [0.22891028, 0.96967952]])
        """
        return np.array([self._feature(candidate) for candidate in range(self.profile_.n_candidates)])

    def plot_features(self, plot_kind="3D", dim=None, row_size=5, show=True):
        """
        This function plot the features vector of
        every candidates in the given dimensions.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``'[0, 1, 2]'``.
        row_size : int
            Number of subplots by row.
            By default, it is set to 5 by rows.
        show : bool
            If True, displays the figure
            at the end of the function.
        """
        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        n_candidate = self.profile_.n_candidates
        n_rows = (n_candidate - 1) // row_size + 1
        fig = plt.figure(figsize=(row_size * 5, n_rows * 5))
        position = [n_rows, row_size, 1]
        features = self.features_
        for candidate in range(n_candidate):
            ax = self.profile_.plot_candidate(candidate,
                                              plot_kind=plot_kind,
                                              dim=dim,
                                              fig=fig,
                                              position=position,
                                              show=False)
            if plot_kind == "3D":
                x1 = features[candidate, dim[0]]
                x2 = features[candidate, dim[1]]
                x3 = features[candidate, dim[2]]
                ax.plot([0, x1], [0, x2], [0, x3], color='k', linewidth=2)
                ax.scatter([x1], [x2], [x3], color='k', s=5)
            elif plot_kind == "ternary":
                x1 = features[candidate, dim[0]]
                x2 = features[candidate, dim[2]]
                x3 = features[candidate, dim[1]]
                feature_bis = [x1, x2, x3]
                feature_bis = np.maximum(feature_bis, 0)
                size_features = np.linalg.norm(feature_bis)
                feature_bis = normalize(feature_bis)
                ax.scatter([feature_bis ** 2], color='k', s=50*size_features+1)
            position[2] += 1

        if show:
            plt.show()  # pragma: no cover


class SVDLog(SVDRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the sum of `log(1 + sigma/C)`
    where sigma are the singular values
    of his embedding matrix and C is a constant.

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.
    const : float
        The constant by which we divide
        the singular values in the log.
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
    [0, 1, 2]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.1637766270779778, 0.0]

    """
    def __init__(self, profile=None, const=1, square_root=True, use_rank=False):
        super().__init__(profile=profile, aggregation_rule=lambda x: np.sum(np.log(1+x/const)),
                         square_root=square_root, use_rank=use_rank)
