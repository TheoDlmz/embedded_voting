from embedded_voting.epistemicGenerators import *
import matplotlib.pyplot as plt
import numpy as np


def test_plot():
    plt.close()
    generator = RatingsGeneratorEpistemicGroupedMean([2, 2, 2], 5, .5)
    generator.plot_scores(show=False)
    plt.close()
    generator = RatingsGeneratorEpistemicMultivariate(np.ones((5, 5)), .5)
    generator.plot_scores(show=False)
    plt.close()

