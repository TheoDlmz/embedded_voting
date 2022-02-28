import numpy as np
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting import RatingsFromEmbeddingsCorrelated
from embedded_voting.manipulation.collective_manipulation.manipulation_coalition import ManipulationCoalition
from embedded_voting.rules.singlewinner_rules.rule_svd_nash import RuleSVDNash
import matplotlib.pyplot as plt


def test_coalition_general():
    plt.clf()
    plt.close("all")
    np.random.seed(42)
    ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    ratings = RatingsFromEmbeddingsCorrelated(coherence=.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    manipulation = ManipulationCoalition(ratings, embeddings, RuleSVDNash())
    manipulation(RuleSVDNash())
    manipulation.manipulation_map(ratings_dim_candidate=np.random.rand(3, 3), show=True)
