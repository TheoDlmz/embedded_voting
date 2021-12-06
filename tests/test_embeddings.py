from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddingsFromRatings import EmbeddingsFromRatingsRandom, EmbeddingsFromRatingsSelf
from embedded_voting.embeddings.generator import EmbeddingsGeneratorRandom, EmbeddingsGeneratorPolarized
import numpy as np
import pytest


def test_embeddings():
    emb = Embeddings(np.array([[.2, .5, .3], [.3, .2, .2], [.6, .2, .3]]))
    emb.dilate(approx=False)
    emb.recenter(approx=False)

    emb = Embeddings(np.array([[1, 1, 1], [1, 1, 1]]))
    emb.dilate()

    emb = Embeddings(np.array([[1, 1]]))
    with pytest.raises(ValueError):
        emb.recenter()
    with pytest.raises(ValueError):
        emb.dilate()

    emb = Embeddings(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    emb.recenter()
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

    Embeddings(np.array([[0, -1], [-1, 0]])).recenter()


def test_embedder():
    ratings = np.array([[1, .8, .5], [.6, .5, .2], [.6, .9, .5]])
    EmbeddingsFromRatingsRandom(n_dim=5)(ratings)
    EmbeddingsFromRatingsSelf()(ratings)


def test_generator():
    EmbeddingsGeneratorRandom(10, 5)()
    with pytest.raises(ValueError):
        EmbeddingsGeneratorPolarized(10, 3)(1.5)
    with pytest.raises(ValueError):
        EmbeddingsGeneratorPolarized(10, 3)(-.5)

