# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.scoring.singlewinner.general import ScoringRule
import matplotlib.pyplot as plt


class FeaturesRule(ScoringRule):
    """
    Voting rule based on the norm of the feature vector of each candidate

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """

    @cached_property
    def features_(self):
        """
        This function return the feature vector of every candidate
        """
        embeddings = self.profile_.embeddings
        scores = self.profile_.scores
        return np.dot(np.dot(np.linalg.pinv(np.dot(embeddings.T, embeddings)), embeddings.T), scores).T

    def score_(self, candidate):
        return (self.features_[candidate] ** 2).sum()

    def plot_features(self, plot_kind="3D", dim=None, row_size=5):
        """
        This function plot the features for every candidate in the given dimensions

        Parameters
        _______
        plot_kind : ["3D", "ternary"]
            the kind of plot we want to show.
        dim : array of length 3
            the three dimensions of the embeddings we want to plot.
            default are [0,1,2]
        row_size : int
            number of figures by row. Default is 5
        """
        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        n_candidate = self.profile_.m
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
                ax.scatter([x1, x2, x3], color='k', s=5)
            elif plot_kind == "ternary":
                x1 = features[candidate, dim[0]]
                x2 = features[candidate, dim[2]]
                x3 = features[candidate, dim[1]]
                feature_bis = [x1, x2, x3]
                feature_bis = np.maximum(feature_bis, 0)
                feature_bis = normalize(feature_bis)
                ax.scatter([feature_bis ** 2], color='k', s=50)
            else:
                raise ValueError("Incorrect value for 'plot_kind'. Should be '3D' or 'ternary'")
            position[2] += 1
        plt.show()
