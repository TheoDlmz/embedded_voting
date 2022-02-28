import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.rules.multiwinner_rules.multiwinner_rule import MultiwinnerRule
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import normalize


class MultiwinnerRuleIter(MultiwinnerRule):
    """
    A class for multi-winner rules
    that are adaptations of STV to the
    embeddings ratings model.

    Parameters
    ----------
    k : int
        The size of the committee.
    quota : str
        The quota used for the re-weighing step.
        Either ``'droop'`` quota `(n/(k+1) +1)` or
        ``'classic'`` quota `(n/k)`.
    take_min : bool
        If True, when the total
        satisfaction is less than the :attr:`quota`,
        we replace the quota by the total
        satisfaction. By default, it is set to False.

    Attributes
    ----------
    quota : str
        The quota used for the re-weighing step.
        Either ``'droop'`` quota `(n/(k+1) +1)` or
        ``'classic'`` quota `(n/k)`.
    take_min : bool
        If True, when the total
        satisfaction is less than the :attr:`quota`,
        we replace the quota by the total
        satisfaction. By default, it is set to False.
    weights: np.ndarray
        Current weight of every voter
    """

    def __init__(self, k=None, quota="classic", take_min=False):
        self.quota = quota
        self.take_min = take_min
        self.weights = np.ones(0)
        if quota not in ["classic", "droop"]:
            raise ValueError("Quota should be either 'classic' (n/k) or 'droop' (n/(k+1) + 1)")
        super().__init__(k=k)

    def _winner_k(self, winners):
        """
        This function determines
        the winner of the k^th iteration.

        Parameters
        ----------
        winners : int list
            The list of the `k-1` first winners in the committee.

        Return
        ------
        int
            The `k^th` winner.
        np.ndarray
            The feature vector associated to this candidate.
            The vector should be of length :attr:`~embedded_voting.Embeddings.embeddings.n_dim`.
        """
        raise NotImplementedError

    def _satisfaction(self, candidate, features_vector):
        """
        This function computes the satisfaction
        of every voter given the winning candidate and
        its features vector.

        Parameters
        ----------
        candidate : int
            The winning candidate.
        features_vector : np.ndarray
            The features vector of the winning candidate.

        Return
        ------
        float list
            The list of the voters' satisfactions
            with this candidate. Should be of
            length :attr:`~embedded_voting.Embeddings.n_voters`.

        """
        temp = [np.dot(self.embeddings.voter_embeddings(i), features_vector) for i in range(self.ratings.n_voters)]
        temp = [self.ratings.voter_ratings(i)[candidate] * temp[i] for i in range(self.ratings.n_voters)]
        return temp

    def _updateWeight(self, satisfactions):
        """
        This function updates voters'
        weights depending on their satisfactions
         with the recently elected candidate.

        Parameters
        ----------
        satisfactions : float list
            The list of the voters' satisfaction
            with this candidate. Should be of
            length :attr:`~embedded_voting.Embeddings.n_voters`.

        Return
        ------
        MultiwinnerRuleIter
            The object itself.
        """
        n_voters = self.ratings.n_voters
        if self.quota == "classic":
            quota_val = n_voters / self.k_
        elif self.quota in "droop":
            quota_val = n_voters / (self.k_ + 1) + 1
        else:
            raise ValueError("Quota should be either 'classic' (n/k) or 'droop' (n/(k+1) + 1)")

        temp = [satisfactions[i] * self.weights[i] for i in range(n_voters)]
        total_sat = np.sum(temp)

        if self.take_min:
            quota_val = min(quota_val, total_sat)

        pond_weights = np.array(temp) * quota_val / total_sat
        self.weights = np.maximum(0, self.weights - pond_weights)
        return self

    def set_quota(self, quota):
        """
        A function to update the :attr:`quota` of the rule.

        Parameters
        ----------
        quota : str
            The new quota, should be
            either ``'droop'`` or ``'classic'``.

        Return
        ------
        MultiwinnerRule
            The object itself.
        """
        if quota not in ["classic", "droop"]:
            raise ValueError("Quota should be either 'classic' (n/k) or 'droop' (n/(k+1) + 1)")
        self.delete_cache()
        self.quota = quota
        return self

    @cached_property
    def _ruleResults(self):
        """
        This function execute the rule and compute
        the winners, their features vectors, and the voters'
        weights at each step.

        Return
        ------
        dict
            A dictionary with 3 elements :
            1) ``winners`` contains the list of winners.
            2) ``vectors`` contains the list of
            candidates features vectors.
            3) ``weights_list`` contains the list of
            voters' weight at each step.
        """
        n_voters = self.ratings.n_voters

        winners = []
        vectors = []
        self.weights = np.ones(n_voters)
        ls_weights = [self.weights]

        for _ in range(self.k_):
            winner_j, vec = self._winner_k(winners)
            vectors.append(vec)
            winners.append(winner_j)

            satisfactions = self._satisfaction(winner_j, vec)

            self._updateWeight(satisfactions)
            ls_weights.append(self.weights)

        return {"winners": winners,
                "vectors": Embeddings(vectors, norm=True),
                "weights_list": ls_weights}

    @cached_property
    def winners_(self):
        """
        This function return the winning committee.

        Return
        ------
        int list
            The winning committee.
        """
        return self._ruleResults["winners"]

    @cached_property
    def features_vectors(self):
        """
        This function return the
        features vectors associated to
        the candidates in the winning committee.

        Return
        ------
        list
            The list of the features vectors of each candidate.
            Each vector is of length :attr:`~embedded_voting.Embeddings.embeddings.n_dim`.

        """
        return self._ruleResults["vectors"]

    def plot_weights(self, plot_kind="3D", dim=None, row_size=5, verbose=True, show=True):
        """
        This function plot the evolution of
        the voters' weights after each step of the rule.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``3D`` or ``ternary``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        row_size : int
            Number of subplots by row.
            By default, it is set to 5.
        verbose : bool
            If True, print the total weight divided by
            the number of remaining candidates at
            the end of each step.
        show : bool
            If True, displays the figure
            at the end of the function.

        """
        ls_weight = self._ruleResults["weights_list"]
        vectors = self._ruleResults["vectors"]
        n_candidates = len(ls_weight)
        n_rows = (n_candidates - 1) // row_size + 1
        fig = plt.figure(figsize=(5 * row_size, n_rows * 5))
        plot_position = [n_rows, row_size, 1]
        if dim is None:
            dim = [0, 1, 2]
        for i in range(n_candidates):
            ax = self.embeddings.plot_ratings_candidate(ls_weight[i],
                                                        plot_kind=plot_kind,
                                                        title="Step %i" % i,
                                                        dim=dim,
                                                        fig=fig,
                                                        plot_position=plot_position,
                                                        show=False)

            if i < n_candidates - 1:
                x1 = vectors[i][dim[0]]
                x2 = vectors[i][dim[1]]
                x3 = vectors[i][dim[2]]
                if plot_kind == "3D":
                    ax.plot([0, x1], [0, x2], [0, x3], color='k', linewidth=2)
                    ax.scatter([x1], [x2], [x3], color='k', s=5)
                elif plot_kind == "ternary":
                    feature_bis = [x1, x3, x2]
                    feature_bis = np.maximum(feature_bis, 0)
                    size_features = np.linalg.norm(feature_bis)
                    feature_bis = normalize(feature_bis)
                    ax.scatter([feature_bis ** 2], color='k', s=50*size_features+1)

            plot_position[2] += 1

        if verbose:
            sum_w = [ls_weight[i].sum() / (n_candidates - i - 1) for i in range(n_candidates - 1)]
            print("Weight / remaining candidate : ", sum_w)

        if show:
            plt.show()  # pragma: no cover

    def plot_winners(self, plot_kind="3D", dim=None, row_size=5, show=True):
        """
        This function plot the winners of the election.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``3D`` or ``ternary``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        row_size : int
            Number of subplots by row.
            By default, it is set to 5.
        show : bool
            If True, displays the figure
            at the end of the function.

        """
        winners = self.winners_
        titles = ["Winner n°%i" % (i + 1) for i in range(self.k_)]

        self.embeddings.plot_candidates(self.ratings,
                                        plot_kind=plot_kind,
                                        dim=dim,
                                        list_candidates=winners,
                                        list_titles=titles,
                                        row_size=row_size,
                                        show=show)
