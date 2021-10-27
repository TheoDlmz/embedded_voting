# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.scoring.singlewinner.general import ScoringRule
from embedded_voting.profile.profile import Profile
from embedded_voting.embeddings.embeddings import Embeddings


class ZonotopeRule(ScoringRule):
    """
    Voting rule in which the aggregated score of
    a candidate is the volume of the Zonotope described by
    his embedding matrix `M` such that `M[i] = score[i, candidate] * embeddings[i]`.
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> embeddings = np.array([[1, 1], [1, 0], [0, 1]])
    >>> profile = Profile(scores, Embeddings(embeddings).normalize())
    >>> election = ZonotopeRule(profile)
    >>> election.scores_
    [(2, 1.016102549694411), (2, 0.5477225575051662), (2, 0.9196152422706632)]
    >>> election.ranking_
    [0, 2, 1]
    >>> election.winner_
    0
    >>> election.welfare_
    [1.0, 0.0, 0.7939978029958996]

    """
    def __init__(self, profile=None):
        super().__init__(profile, _score_components=2)

    def _score_(self, candidate):
        embeddings = self.profile_.scored_embeddings(candidate)
        matrix_rank = np.linalg.matrix_rank(embeddings)
        volume = 0
        n_voters = self.profile_.n_voters
        current_subset = list(np.arange(matrix_rank))
        while current_subset[0] <= n_voters - matrix_rank:
            current_embeddings = embeddings[current_subset, ...]
            if matrix_rank < self.profile_.embeddings.n_dim:
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
    Voting rule in which the aggregated score of
    a candidate is the volume of a cube
    described by :attr:`~embedded_voting.Profile.embeddings.n_dim` rows of
    the candidate embedding matrix `M` such
    that `M[i] = score[i, candidate] * embeddings[i]`.
    (cf :meth:`~embedded_voting.Profile.scored_embeddings`).

    Parameters
    ----------
    profile: Profile
        The profile of voters on which we run the election.

    Examples
    --------
    >>> scores = np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]])
    >>> embeddings = np.array([[1, 1], [1, 0], [0, 1]])
    >>> profile = Profile(scores, Embeddings(embeddings).normalize())
    >>> election = MaxCubeRule(profile)
    >>> election.scores_
    [(2, 0.4183300132670379), (2, 0.5477225575051662), (2, 0.3999999999999999)]
    >>> election.ranking_
    [1, 0, 2]
    >>> election.winner_
    1
    >>> election.welfare_
    [0.12408405037529173, 1.0, 0.0]

    """
    def __init__(self, profile=None):
        super().__init__(profile, _score_components=2)

    def _score_(self, candidate):
        embeddings = self.profile_.scored_embeddings(candidate)
        matrix_rank = np.linalg.matrix_rank(embeddings)
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
