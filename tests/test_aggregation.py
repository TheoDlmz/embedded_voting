from embedded_voting.algorithm_aggregation.score_generator import GroupedMeanGenerator, MultivariateGenerator
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

