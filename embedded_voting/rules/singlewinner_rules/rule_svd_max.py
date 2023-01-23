import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule_svd import RuleSVD
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import normalize


class RuleSVDMax(RuleSVD):
    """
    Voting rule in which the aggregated score of a candidate is the maximum singular value
    of his embedding matrix (cf :meth:`~embedded_voting.Embeddings.times_ratings_candidate`).

    Parameters
    ----------
    square_root: boolean
        If True, use the square root of score in the matrix.
        By default, it is True.
    use_rank : boolean
        If True, consider the rank of the matrix when doing the ranking.
        By default, it is False.
    embedded_from_ratings: EmbeddingsFromRatings
        If no embeddings are specified in the call, this `EmbeddingsFromRatings` object is use to generate
        the embeddings from the ratings. Default: `EmbeddingsFromRatingsIdentity()`.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
    >>> election = RuleSVDMax()(ratings, embeddings)
    >>> election.scores_  # DOCTEST: +ELLIPSIS
    [1.0264274892038..., 1.1760506747094..., 0.9926782946277...]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_  # DOCTEST: +ELLIPSIS
    [0.184047317055..., 1.0, 0.0]

    """
    def __init__(self, square_root=True, use_rank=False, embedded_from_ratings=None):
        super().__init__(aggregation_rule=np.max, square_root=square_root, use_rank=use_rank,
                         embedded_from_ratings=embedded_from_ratings)

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
            candidate, of length :attr:`~embedded_voting.Embeddings.n_dim`.
        """
        if self.square_root:
            m_candidate = self.embeddings_.times_ratings_candidate(np.sqrt(self.ratings_[::, candidate]))
        else:
            m_candidate = self.embeddings_.times_ratings_candidate(self.ratings_[::, candidate])
        _, vp, vec = np.linalg.svd(m_candidate)
        vec = vec[0]
        if vec.sum() < 0:
            return - vec * vp[0]
        else:
            return vec * vp[0]

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
            of shape :attr:`~embedded_voting.Ratings.n_candidates`, :attr:`~embedded_voting.Embeddings.n_dim`.

        Examples
        --------
        >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
        >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
        >>> election = RuleSVDMax()(ratings, embeddings)
        >>> election.features_
        array([[0.94829535, 0.39279679],
               [0.31392742, 1.13337759],
               [0.22807074, 0.96612315]])
        """
        return np.array([self._feature(candidate) for candidate in range(self.ratings_.n_candidates)])

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

        n_candidate = self.ratings_.shape[1]
        n_rows = (n_candidate - 1) // row_size + 1
        fig = plt.figure(figsize=(row_size * 5, n_rows * 5))
        plot_position = [n_rows, row_size, 1]
        features = self.features_
        for candidate in range(n_candidate):
            ax = self.embeddings_.plot_candidate(self.ratings_,
                                                 candidate,
                                                 plot_kind=plot_kind,
                                                 dim=dim,
                                                 fig=fig,
                                                 plot_position=plot_position,
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
            plot_position[2] += 1

        if show:
            plt.show()  # pragma: no cover
