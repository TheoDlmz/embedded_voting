import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.ratings.ratings_generator import RatingsGenerator
from embedded_voting.truth.truth_generator import TruthGenerator
from embedded_voting.truth.truth_generator_uniform import TruthGeneratorUniform


class RatingsGeneratorEpistemic(RatingsGenerator):
    """
    A generator of ratings based on a ground truth ("true value") for each candidate.

    Parameters
    ----------
    n_voters : int
        The number of voters in the generator.
    truth_generator : TruthGenerator
        The truth generator used to generate to true values of each candidate.
        Default: `TruthGeneratorUniform(10, 20)`.

    Attributes
    ----------
    ground_truth_ : np.ndarray
        The ground truth ("true value") for each candidate, corresponding to the
        last ratings generated.
    """

    def __init__(self, n_voters=None, truth_generator=None):
        if truth_generator is None:
            truth_generator = TruthGeneratorUniform(minimum_value=10, maximum_value=20)
        self.truth_generator = truth_generator
        super().__init__(n_voters)
        self.ground_truth_ = None

    def __call__(self, n_candidates=1):
        """
        This function generate the ground truth and the ratings given by each voter to
        n_candidates candidates.

        Parameters
        ----------
        n_candidates : int
            The number of candidates of which we want the ratings.

        Return
        ------
        Ratings
            The ratings given by the voters to the candidates.
        """
        raise NotImplementedError

    def plot_ratings(self, show=True):
        """
        This function plots the true value of a candidate and the ratings
        given by each voter for a candidate with new random values and ratings.

        Parameters
        ----------
        show : bool
            If True, displays the plot at the end of the function.
        """
        ratings = self()
        fig, ax = plt.subplots()
        ax.plot([self.ground_truth_[0]] * 2, [0, 1], color="red", label="True value")
        self._plot_ratings_aux(ax=ax, ratings=ratings)
        ax.set_ylim(0, 1)
        ax.set_title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()  # pragma: no cover

    def _plot_ratings_aux(self, ax, ratings):
        """
        Auxiliary method for the plot itself (without boilerplate code like title, legend, etc).

        Parameters
        ----------
        ax : ax
            The matplotlib ax object.
        ratings : Ratings
            Ratings of the candidate.
        """
        for i_voter in range(self.n_voters):
            ax.plot([ratings[i_voter]] * 2, [0, 1], color="k")
