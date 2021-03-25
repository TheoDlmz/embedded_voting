from embedded_voting.algorithm_aggregation.score_generator import GroupedMeanGenerator, MultivariateGenerator
from embedded_voting.algorithm_aggregation.auto_embeddings import AutoProfile
import matplotlib.pyplot as plt
import numpy as np


def test_plot():
    plt.close()
    generator = GroupedMeanGenerator([2, 2, 2], 5, .5)
    generator.plot_scores(show=False)
    plt.close()
    generator = MultivariateGenerator(np.ones((5, 5)), .5)
    generator.plot_scores(show=False)
    plt.close()


def test_auto_profile():
    np.random.seed(42)
    profile = AutoProfile(20, 4)
    profile.add_voters_auto(np.random.rand(10, 20), np.random.rand(10, 100))
    profile.add_voters_cov(np.random.rand(10, 20), np.random.rand(10, 100), normalize_score=True)
