from embedded_voting.ratings import *
import matplotlib.pyplot as plt
import numpy as np


def test_plot():
    plt.close()
    generator = RatingsGeneratorEpistemicGroupsMean([2, 2, 2], 5, .5)
    generator.plot_ratings(show=False)
    plt.close()
    generator = RatingsGeneratorEpistemicMultivariate(np.ones((5, 5)), .5)
    generator.plot_ratings(show=False)
    plt.close()

