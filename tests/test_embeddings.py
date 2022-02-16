from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_random import EmbeddingsFromRatingsRandom
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_self import EmbeddingsFromRatingsSelf
from embedded_voting.embeddings.embeddings_generator_uniform import EmbeddingsGeneratorUniform
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
import numpy as np
import pytest


def test_embeddings():
    emb = Embeddings(np.array([[.2, .5, .3], [.3, .2, .2], [.6, .2, .3]]), norm=True)
    emb.dilated(approx=False)
    emb.recentered(approx=False)

    emb = Embeddings(np.array([[1, 1, 1], [1, 1, 1]]), norm=True)
    emb.dilated()

    emb = Embeddings(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), norm=True)
    emb.recentered()
    emb.plot("3D", show=False)
    emb.plot("ternary", dim=[1, 2, 0], show=False)
    with pytest.raises(ValueError):
        emb.plot("test", show=False)
    with pytest.raises(ValueError):
        emb.plot("3D", dim=[1, 2], show=False)

    ratings = np.array([[1, .8, .5], [.6, .5, .2], [.6, .9, .5]])
    emb.plot_candidate(ratings, 0, "3D", show=False)
    emb.plot_candidate(ratings, 0, "ternary", dim=[1, 2, 0], show=False)
    with pytest.raises(ValueError):
        emb.plot_candidate(ratings, 0, "test", show=False)

    emb.plot_candidates(Ratings(ratings), "3D", show=False)
    emb.plot_candidates(ratings, "ternary", show=False, list_titles=["c_1", "c_2", "c_3"])
    with pytest.raises(ValueError):
        emb.plot_candidates(ratings, "test", show=False)

    Embeddings(np.array([[0, -1], [-1, 0]]), norm=True).recentered()


def test_embeddings_get_center():
    """
    >>> embeddings = Embeddings([[1, 0], [.7, .7], [0, 1]], norm=True)
    >>> embeddings.get_center()
    array([0.70710678, 0.70710678])
    >>> embeddings = Embeddings([[1, 0], [0, 1], [.7, .7]], norm=True)
    >>> embeddings.get_center()
    array([0.70710678, 0.70710678])
    """
    pass


def test_embeddings_copy():
    """
    >>> embeddings = Embeddings(np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]]), norm=False)
    >>> embeddings.some_attribute = 51
    >>> second_embeddings = embeddings.copy()
    >>> second_embeddings
    Embeddings([[0.5, 0.9, 0.4],
                [0.4, 0.7, 0.5],
                [0.4, 0.2, 0.4]])
    >>> second_embeddings.some_attribute
    51
    >>> second_embeddings[0, 0] = 42
    >>> second_embeddings
    Embeddings([[42. ,  0.9,  0.4],
                [ 0.4,  0.7,  0.5],
                [ 0.4,  0.2,  0.4]])
    >>> embeddings
    Embeddings([[0.5, 0.9, 0.4],
                [0.4, 0.7, 0.5],
                [0.4, 0.2, 0.4]])
    """
    pass


def test_embedder():
    ratings = np.array([[1, .8, .5], [.6, .5, .2], [.6, .9, .5]])
    EmbeddingsFromRatingsRandom(n_dim=5)(ratings)
    EmbeddingsFromRatingsSelf(norm=True)(ratings)
