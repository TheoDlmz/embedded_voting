
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from embedded_voting.ratings.ratingsGenerator import RatingsGenerator


class RatingsGeneratorEpistemic(RatingsGenerator):
    """
    A generator of scores for a group of algorithms.

    Parameters
    ----------
    n_voters : int
        The number of voters in the generator.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.

    Attributes
    ----------
    n_voters : int
        The number of voters in the generator.
    minimum_score : float or int
        The minimum true value of an alternative.
        By default, it is set to 10.
    maximum_score : float or int
        The maximum true value of an alternative.
        By default, it is set to 20.
    groups_sizes : list
        The number of voter in each group.
        If set to None, then there is no "groups".

    """

    def __init__(self, n_voters=1, minimum_score=10, maximum_score=20):
        super().__init__(n_voters)
        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.groups_sizes = None

    def generate_true_score(self):
        """
        This function generate a true value for some alternative.

        Return
        ------
        float
            The true value of an alternative
        """
        return self.minimum_score + np.random.rand()*(self.maximum_score-self.minimum_score)

    def __call__(self, n_candidates=1, *args):
        """
        This function computes the true values and the scores given by each voter to
        different alternative.

        Parameters
        ----------
        n_candidates : int
            The number of candidates of which we want the score.

        Return
        ------
        np.ndarray
            A list of length ``n_candidates`` containing the true values of each alternative.
        Ratings
            The ratings given by the voters to the alternatives
        """
        raise NotImplementedError

    def plot_scores(self, show=True):
        """
        This function plots the true value of a candidate and the scores
        given by each voter for some candidate randomly selected.

        Parameters
        ----------
        show : bool
            If True, displays the plot at the end of the function.
        """
        true_value, scores = self()
        fig, ax = plt.subplots()
        if self.groups_sizes is not None:
            color = cm.rainbow(np.linspace(0, 0.8, len(self.groups_sizes)))
            count = 0
            n_group = -1
        else:
            color = ["k"]
            count = self.n_voters
            n_group = 0
        ax.plot([true_value[0]]*2, [0, 1], color="red", label="True value")
        for i in range(self.n_voters):
            if i >= count:
                count += self.groups_sizes[n_group+1]
                n_group += 1
                ax.plot([scores[i]]*2, [0, 1], color=color[n_group], label="group %i" % (n_group+1))
            else:
                ax.plot([scores[i]]*2, [0, 1], color=color[n_group])
        ax.set_ylim(0, 1)
        ax.set_title("Distribution of voters' guesses")
        plt.legend()
        if show:
            plt.show()  # pragma: no cover

