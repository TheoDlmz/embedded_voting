import numpy as np
from embedded_voting.embeddings.generator import ParametrizedEmbeddings
from embedded_voting.profile.generator import CorrelatedRatings
from embedded_voting.manipulation.coalition.general import ManipulationCoalition
from embedded_voting.scoring.singlewinner.svd import SVDNash
import matplotlib.pyplot as plt


def test_coalition_general():
    plt.clf()
    plt.close("all")
    np.random.seed(42)
    scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    embeddings = ParametrizedEmbeddings(10, 3)(.8)
    ratings = CorrelatedRatings(3, 3, scores_matrix)(embeddings, .8)
    manipulation = ManipulationCoalition(ratings, embeddings, SVDNash())
    manipulation(SVDNash())
    manipulation.manipulation_map(scores_matrix=np.random.rand(3, 3), show=True)
