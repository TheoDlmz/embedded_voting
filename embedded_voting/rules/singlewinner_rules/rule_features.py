# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import normalize


class RuleFeatures(Rule):
    """
    Voting rule in which the aggregated score of a candidate is the norm of the feature vector of this candidate.

    Intuitively, for each candidate, her feature on embedding dimension `d` is the ideal rating that a voter of
    group `d` should put to that candidate. In this model, the actual rating of a voter for this candidate would be
    a mean of the features, weighted by the voter's embedding: `embeddings[voter, :] @ features[candidate, :]`.
    Considering all the voters and all the candidates, we then obtain `ratings = embeddings @ features.T`, i.e.
    `features = (inv(embeddings) @ ratings).T`.

    Since `embeddings` is not always invertible, we consider in practice `features = (pinv(embeddings) @ ratings).T`.
    This can be seen as a least-square approximation of the inital model.

    Finally, the score of a candidate is the Euclidean norm of her vector of features.

    Examples
    --------
    >>> ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    >>> embeddings = Embeddings(np.array([[1, 1], [1, 0], [0, 1]]), norm=True)
    >>> election = RuleFeatures()(ratings, embeddings)
    >>> election.scores_
    [0.669..., 0.962..., 0.658...]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.0353..., 1.0, 0.0]
    """

    @cached_property
    def features_(self):
        """
        This function return the feature vector of all candidates.

        Return
        ------
        np.ndarray
            The matrix of features.
            Its shape is :attr:`~embedded_voting.Ratings.n_candidates`, :attr:`~embedded_voting.Embeddings.n_dim`.
        """
        return np.array((np.linalg.pinv(self.embeddings_) @ self.ratings_).T)

    def _score_(self, candidate):
        return np.linalg.norm(self.features_[candidate])

    def plot_features(self, plot_kind="3D", dim: list = None, row_size=5, show=True):
        """
        This function plot the features vector of
        all candidates in the given dimensions.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        row_size : int
            The number of subplots by row.
            By default, it is set to 5 plots by row.
        show : bool
            If True, plot the figure
            at the end of the function.
        """
        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        n_candidates = self.ratings_.n_candidates
        n_rows = (n_candidates - 1) // row_size + 1
        fig = plt.figure(figsize=(row_size * 5, n_rows * 5))
        plot_position = [n_rows, row_size, 1]
        features = self.features_
        for candidate in range(n_candidates):
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
