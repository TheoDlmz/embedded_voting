# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringFunction
import matplotlib.pyplot as plt

class FeaturesRule(ScoringFunction):
    """
    Voting rule based on the norm of the feature vector of each candidate

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """

    @cached_property
    def features_(self):
        X = self.profile_.embs
        S = self.profile_.scores
        try:
            return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), S).T
        except:
            return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), S).T

    def features_quality(self):
        f = self.features_
        X = self.profile_.embs
        S = self.profile_.scores
        diff = np.dot(X, f.T) - S
        diff = diff ** 2
        diff = diff.mean(axis=0)
        return diff

    def score_(self, cand):
        return (self.features_[cand] ** 2).sum()

    def plot_winner(self, verbose=False, space="3D"):
        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=[self.winner_],
                                        list_titles=["Winner with sum of square of features"])
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=[self.winner_],
                                        list_titles=["Winner with sum of square of features"])
        else:
            raise ValueError("Incorrect space value (3D/2D)")

    def plot_features(self, space="3D"):
        n_cand = self.profile_.m
        n_rows = (n_cand - 1) // 6 + 1
        fig = plt.figure(figsize=(30, n_rows * 5))
        intfig = [n_rows, 6, 1]
        f = self.features_
        for cand in range(n_cand):
            if space == "3D":
                ax = self.profile_.plot_scores_3D(self.profile_.scores[::, cand],
                                                  title="Candidate %i" % (cand + 1),
                                                  fig=fig,
                                                  intfig=intfig,
                                                  show=False)
                ax.plot([0, f[cand, 0]], [0, f[cand, 1]], [0, f[cand, 2]], color='k', linewidth=2)
                ax.scatter([f[cand, 0]], [f[cand, 1]], [f[cand, 2]], color='k', s=5)
            elif space == "2D":
                ax = self.profile_.plot_scores_2D(self.profile_.scores[::, cand],
                                                  title="Candidate %i" % (cand + 1),
                                                  show=False)

                fbis = np.maximum(f[cand], 0)
                fbis = fbis / np.linalg.norm(fbis)
                ax.scatter([fbis ** 2], color='k', s=50)
                plt.show()
            else:
                raise ValueError("Incorrect value for space (3D/2D)")

            intfig[2] += 1

        plt.show()
