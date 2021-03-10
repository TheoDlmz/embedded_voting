# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule


class ZonotopeRule(ScoringRule):
    """
    Voting rule that rank candidates by the volume of the Zonotope described by
    the embedding matrix using voters' scores for the candidate.

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """
    def __init__(self, profile):
        super().__init__(profile)
        self.score_components = 2

    def score_(self, candidate):
        embeddings = self.profile_.scored_embeddings(candidate)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        volume = 0
        n_voters = self.profile_.n_voters
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n_voters - matrix_rank:
            current_embeddings = embeddings[current_subset, ...]
            if matrix_rank < self.profile_.n_dim:
                vol_i = np.linalg.det(np.dot(current_embeddings, current_embeddings.T))
                vol_i = max(0, vol_i)
                vol_i = np.sqrt(vol_i)
            else:
                vol_i = np.linalg.det(current_embeddings)
                vol_i = np.abs(vol_i)
            volume += vol_i
            x = 1
            while current_subset[matrix_rank - x] == n_voters - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return matrix_rank, volume


class MaxCubeRule(ScoringRule):
    """
    Voting rule that rank candidates by the volume of the maximum volume cube described by a subset
    of dim (the number of dimensions) voters

    Parameters
    _______
    profile: Profile
        the profile of voter on which we run the election

    """
    def __init__(self, profile):
        super().__init__(profile)
        self.score_components = 2

    def score_(self, candidate):
        embeddings = self.profile_.scored_embeddings(candidate)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        if matrix_rank == 0:
            return 0, 0

        volume = 0
        n_voters = self.profile_.n_voters
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n_voters - matrix_rank:
            current_embeddings = embeddings[current_subset, ...]
            volume = max(volume, np.sqrt(np.linalg.det(np.dot(current_embeddings, current_embeddings.T))))
            x = 1
            while current_subset[matrix_rank - x] == n_voters - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return matrix_rank, volume
