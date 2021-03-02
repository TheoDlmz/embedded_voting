# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.utils.cached import cached_property
from embedded_voting.scoring.singlewinner.general import ScoringFunction


class ZonotopeRule(ScoringFunction):
    """
    Voting rule that rank candidates by the volume of the Zonotope described by
    the embedding matrix using voters' scores for the candidate.

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """
    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        volume = 0
        n = self.profile_.n
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n - matrix_rank:
            E = embeddings[current_subset, ...]
            if matrix_rank < self.profile_.dim:
                vol_i = np.linalg.det(np.dot(E, E.T))
                vol_i = max(0, vol_i)
                vol_i = np.sqrt(vol_i)
            else:
                vol_i = np.linalg.det(E)
                vol_i = np.abs(vol_i)
            volume += vol_i
            x = 1
            while current_subset[matrix_rank - x] == n - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return (matrix_rank, volume)

    @cached_property
    def ranking_(self):
        rank = [s[0] for s in self.scores_]
        scores = [s[1] for s in self.scores_]
        return np.lexsort((scores, rank))[::-1]

    def plot_winner(self, space="3D"):
        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=[self.winner_], list_titles=["ZonotopeRule Winner"])
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=[self.winner_], list_titles=["ZonotopeRule Winner"])
        else:
            raise ValueError("Incorrect space value (3D/2D)")


class maxCubeRule(ScoringFunction):
    """
    Voting rule that rank candidates by the volume of the maximum volume cube described by a subset
    of dim (the number of dimensions) voters

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """
    def score_(self, cand):
        embeddings = self.profile_.scored_embeddings(cand)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        if matrix_rank == 0:
            return (0, 0)

        # dim = self.profile_.dim
        volume = 0
        n = self.profile_.n
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n - matrix_rank:
            E = embeddings[current_subset, ...]
            volume = max(volume, np.sqrt(np.linalg.det(np.dot(E, E.T))))
            x = 1
            while current_subset[matrix_rank - x] == n - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return (matrix_rank, volume)

    @cached_property
    def ranking_(self):
        rank = [s[0] for s in self.scores_]
        scores = [s[1] for s in self.scores_]
        return np.lexsort((scores, rank))[::-1]

    def plot_winner(self, space="3D"):
        if space == "3D":
            self.profile_.plot_cands_3D(list_cand=[self.winner_], list_titles=["MaxCubeRule Winner"])
        elif space == "2D":
            self.profile_.plot_cands_2D(list_cand=[self.winner_], list_titles=["MaxCubeRule Winner"])
        else:
            raise ValueError("Incorrect space value (3D/2D)")
