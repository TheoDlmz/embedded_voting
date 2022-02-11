import numpy as np
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting import RatingsFromEmbeddingsCorrelated
from embedded_voting.manipulation.coalition.general import ManipulationCoalition
from embedded_voting.scoring.singlewinner.svd import SVDNash
import matplotlib.pyplot as plt


def test_coalition_general():
    plt.clf()
    plt.close("all")
    np.random.seed(42)
    ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    ratings = RatingsFromEmbeddingsCorrelated(3, 3, .8, ratings_dim_candidate)(embeddings)
    manipulation = ManipulationCoalition(ratings, embeddings, SVDNash())
    manipulation(SVDNash())
    manipulation.manipulation_map(ratings_dim_candidate=np.random.rand(3, 3), show=True)
