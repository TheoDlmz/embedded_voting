# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.utils.plots import create_ternary_plot, create_3D_plot


class Embeddings(np.ndarray):
    """
    Embeddings of voters

    Parameters
    ----------
    positions : np.ndarray or list or Embeddings
        The embeddings of the voters. Its dimensions are :attr:`n_voters`, :attr:`n_dim`.
    norm: bool
        If True, normalize the embeddings

    Attributes
    ----------
    n_voters : int
        The number of voters in the ratings.
    n_dim : int
        The number of dimensions of the voters' embeddings.

    Examples
    --------
    >>> embs = Embeddings([[1, 0], [0, 1], [0.5, 0.5]])
    >>> embs.n_voters
    3
    >>> embs.n_dim
    2
    >>> embs.voter_embeddings(0)
    array([1., 0.])
    """

    def __new__(cls, positions, norm=True):
        obj = np.asarray(positions).view(cls)
        if norm:
            obj = (obj.T / np.sqrt((obj ** 2).sum(axis=1))).T
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if len(self.shape) == 2:
            self.n_dim, self.n_voters = self.shape

    def voter_embeddings(self, i):
        return np.array(self[i:i + 1, :])[0]

    def scored(self, ratings):
        """
        This method compute the embeddings multiplied by ratings given by the voter. For each voter, its
        embeddings are multiplied by the given rating

        Parameters
        ----------
        ratings: np.ndarray
            The vector of ratings given by the voters

        Return
        ------
        np.ndarray
            The scored embeddings

        Examples
        --------
        >>> embs = Embeddings(np.array([[1, 0], [0, 1], [0.5, 0.5]]), norm=False)
        >>> embs.scored(np.array([.8, .5, .4]))
        Embeddings([[0.8, 0. ],
                    [0. , 0.5],
                    [0.2, 0.2]])
        """
        return np.multiply(self, ratings[::, np.newaxis])

    def _get_center(self):
        """
        Return the center of the ratings, computed
        as the center of the :attr:`n_dim`-dimensional
        cube of maximal volume.

        Return
        ------
        np.ndarray
            The position of the center vector. Should be of length :attr:`n_dim`.
        """

        positions = np.array(self)
        matrix_rank = np.linalg.matrix_rank(positions)
        volume = 0
        n_voters = self.n_voters
        current_subset = list(np.arange(matrix_rank))
        mean = np.zeros(self.n_dim)
        while current_subset[0] <= n_voters - matrix_rank:
            current_embeddings = positions[current_subset, ...]
            new_volume = np.sqrt(np.linalg.det(np.dot(current_embeddings, current_embeddings.T)))
            if new_volume > volume:
                volume = new_volume
                mean = normalize(current_embeddings.sum(axis=0))
            x = 1
            while current_subset[matrix_rank - x] == n_voters - x:
                x += 1
            val = current_subset[matrix_rank - x] + 1
            while x > 0:
                current_subset[matrix_rank - x] = val
                val += 1
                x -= 1

        return mean

    def dilate(self, approx=True):
        """
        Dilate the embeddings of the
        voters so that they take all
        the space possible in the non-negative orthant.

        Parameters
        ----------
        approx : bool
            If True, we compute the center of the population
            with a polynomial time algorithm. If False, we use
            an algorithm exponential in :attr:`n_dim`.

        Return
        ------
        Embeddings
            The embeddings itself.

        Examples
        --------
        >>> embs = Embeddings(np.array([[.5,.4,.4],[.4,.4,.5],[.4,.5,.4]])).normalize()
        >>> embs
        Embeddings([[0.66226618, 0.52981294, 0.52981294],
                    [0.52981294, 0.52981294, 0.66226618],
                    [0.52981294, 0.66226618, 0.52981294]])
        >>> embs.dilate()
        Embeddings([[0.98559856, 0.11957316, 0.11957316],
                    [0.11957316, 0.11957316, 0.98559856],
                    [0.11957316, 0.98559856, 0.11957316]])
        """

        if self.n_voters < 2:
            raise ValueError("Cannot dilate a ratings with less than 2 candidates")

        positions = np.array(self)
        if approx:
            center = normalize(positions.sum(axis=0))
        else:
            center = self._get_center()

        min_value = np.dot(self.voter_embeddings(0), center)
        for i in range(self.n_voters):
            val = np.dot(self.voter_embeddings(i), center)
            if val < min_value:
                min_value = val

        theta_max = np.arccos(min_value)
        k = np.pi / (4 * theta_max)

        new_positions = np.zeros((self.n_voters, self.n_dim))
        for i in range(self.n_voters):
            v_i = self.voter_embeddings(i)
            theta_i = np.arccos(np.dot(v_i, center))
            if theta_i == 0:
                new_positions[i] = v_i
            else:
                p_1 = np.dot(center, v_i) * center
                p_2 = v_i - p_1
                e_2 = normalize(p_2)
                new_positions[i] = center * np.cos(k * theta_i) + e_2 * np.sin(k * theta_i)

        return Embeddings(new_positions, False)

    def recenter(self, approx=True):
        """
        Recenter the embeddings of the
        voters so that they are the most
        possible on the non-negative orthant.

        Parameters
        ----------
        approx : bool
            If True, we compute the center of the population
            with a polynomial time algorithm. If False, we use
            an algorithm exponential in :attr:`n_dim`.

        Return
        ------
        Embeddings
            The embeddings itself.

        Examples
        --------
        >>> embs = Embeddings(-np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]])).normalize()
        >>> embs
        Embeddings([[-0.45267873, -0.81482171, -0.36214298],
                    [-0.42163702, -0.73786479, -0.52704628],
                    [-0.66666667, -0.33333333, -0.66666667]])
        >>> embs.recenter()
        Embeddings([[0.40215359, 0.75125134, 0.52334875],
                    [0.56352875, 0.6747875 , 0.47654713],
                    [0.70288844, 0.24253193, 0.66867489]])
        """

        if self.n_voters < 2:
            raise ValueError("Cannot recenter a ratings with less than 2 candidates")

        positions = np.array(self)
        if approx:
            center = normalize(positions.sum(axis=0))
        else:
            center = self._get_center()

        target_center = np.ones(self.n_dim)
        target_center = normalize(target_center)
        if np.dot(center, target_center) == -1:
            new_positions = - self
        elif np.dot(center, target_center) == 1:
            new_positions = self
        else:
            orthogonal_center = center - np.dot(center, target_center)*target_center
            orthogonal_center = normalize(orthogonal_center)
            theta = -np.arccos(np.dot(center, target_center))
            rotation_matrix = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            new_positions = np.zeros((self.n_voters, self.n_dim))
            for i in range(self.n_voters):
                position_i = self.voter_embeddings(i)
                comp_1 = position_i.dot(target_center)
                comp_2 = position_i.dot(orthogonal_center)
                vector = [comp_1, comp_2]
                remainder = position_i - comp_1*target_center - comp_2*orthogonal_center
                new_vector = rotation_matrix.dot(vector)
                new_positions[i] = new_vector[0]*target_center + new_vector[1]*orthogonal_center + remainder

        return Embeddings(new_positions, False)

    def normalize(self):
        """
        Normalize the embeddings of the
        voters so the norm of every embedding is 1.

        Return
        ------
        Embeddings
            The embeddings itself.

        Examples
        --------
        >>> embs = Embeddings(-np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]]), norm=False)
        >>> embs
        Embeddings([[-0.5, -0.9, -0.4],
                    [-0.4, -0.7, -0.5],
                    [-0.4, -0.2, -0.4]])
        >>> embs.normalize()
        Embeddings([[-0.45267873, -0.81482171, -0.36214298],
                    [-0.42163702, -0.73786479, -0.52704628],
                    [-0.66666667, -0.33333333, -0.66666667]])
        """
        new_positions = (self.T / np.sqrt((self ** 2).sum(axis=1))).T
        return Embeddings(new_positions, False)

    def _plot_3D(self, fig, dim, plot_position=None):
        """
        Plot a figure of the ratings
        on a 3D space using matplotlib.

        Parameters
        ----------
        fig : matplotlib.figure
            The figure on which we do the plot.
        dim : list
            The 3 dimensions we are using for our plot.
        plot_position : list
            The position of the plot on the figure.
            Should be of the form
             ``[n_rows, n_columns, position]``.

        Return
        ------
        matplotlib.ax
            The matplotlib ax
            with the figure, if
            you want to add
            something to it.

        """
        ax = create_3D_plot(fig, plot_position)
        for i, v in enumerate(self):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            ax.plot([0, x1], [0, x2], [0, x3], color=(x1**2 * 0.8, x2**2 * 0.8, x3**2 * 0.8), alpha=0.4)
            ax.scatter([x1], [x2], [x3], color='k', s=1)
        return ax

    def _plot_ternary(self, fig, dim, plot_position=None):
        """
        Plot a figure of the ratings on a 2D space
        representing the surface of the unit sphere
        on the non-negative orthant.

        Parameters
        ----------
        fig : matplotlib figure
            The figure on which we add the plot.
        dim : list
            The 3 dimensions we are using for our plot.
        plot_position : list
            The position of the plot on the figure.
            Should be of the form
            ``[n_rows, n_columns, position]``.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure,
            if you want to add something to it.

        """
        tax = create_ternary_plot(fig, plot_position)
        for i, v in enumerate(self):
            x1 = v[dim[0]]
            x2 = v[dim[2]]
            x3 = v[dim[1]]
            vec = [x1, x2, x3]
            tax.scatter([normalize(vec)**2], color=(x1**2 * 0.8, x3**2 * 0.8, x2**2 * 0.8), alpha=0.9, s=30)

        return tax

    def plot(self, plot_kind="3D", dim=None, fig=None, plot_position=None, show=True):
        """
        Plot the ratings of the voters,
        either on a 3D plot, or on a ternary plot.
        Only three dimensions can be represented.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            A list of length 3 containing
            the three dimensions of the
            embeddings we want to plot.
            All elements of this list should
            be lower than :attr:`n_dim`.
            By default, it is set to ``[0, 1, 2]``.
        fig : matplotlib figure
            The figure on which we add the plot.
            The default figure is a
            `10 x 10` matplotlib figure.
        plot_position : list
            List of length 3 containing the
            matplotlib position ``[n_rows, n_columns, position]``.
            By default, it is set to ``[1, 1, 1]``.
        show : bool
            If True, display the figure at
            the end of the function.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure,
            if you want to add something to it.

        """

        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        if plot_kind == "3D":
            ax = self._plot_3D(fig, dim, plot_position)
        elif plot_kind == "ternary":
            ax = self._plot_ternary(fig, dim, plot_position)
        else:
            raise ValueError("plot_kind should '3D' or 'ternary'")
        ax.set_title("Profile of voters (%i,%i,%i)" % (dim[0], dim[1], dim[2]), fontsize=24)
        if show:
            plt.show()  # pragma: no cover
        return ax

    def _plot_scores_3D(self, sizes, fig, plot_position, dim):
        """
        Plot a figure of the ratings on a
        3D space with the embeddings vector
        having the sizes passed as parameters.

        Parameters
        ----------
        sizes : np.ndarray
            The norm of the vectors.
            Should be of length :attr:`n_voters`.
        fig : matplotlib figure
            The figure on which we add the plot.
        plot_position : list
            The position of the plot on the figure.
            Should be of the form
            ``[n_rows, n_columns, position]``.
        dim : list
            The 3 dimensions we are using for our plot.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure,
            if you want to add something to it.

        """
        ax = create_3D_plot(fig, plot_position)
        for i, (v, s) in enumerate(zip(np.array(self), sizes)):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            ax.plot([0, s * x1], [0, s * x2], [0, s * x3], color=(x1**2 * 0.8, x2**2 * 0.8, x3**2 * 0.8), alpha=0.4)

        return ax

    def _plot_scores_ternary(self, sizes, fig, plot_position, dim):
        """
        Plot a figure of the ratings on a 2D space
        representing the sphere in the non-negative orthant,
        with the voters dots having the sizes passed
        as parameters.

        Parameters
        ----------
        sizes : np.ndarray
            The size of the dots.
            Should be of length :attr:`n_voters`.
        fig : matplotlib figure
            The figure on which we add the plot.
        plot_position : list
            The position of the plot on the figure. Should be of the form
            ``[n_rows, n_columns, position]``.
        dim : list
            The 3 dimensions we are using for our plot.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure,
            if you want to add something to it.

        """
        tax = create_ternary_plot(fig, plot_position)
        for i, (v, s) in enumerate(zip(np.array(self), sizes)):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            vec = [x1, x2, x3]
            tax.scatter([normalize(vec)**2], color=(x1**2 * 0.8, x3**2 * 0.8, x2**2 * 0.8), alpha=0.7, s=max(s * 50, 1))

        return tax

    def plot_scores(self, sizes, title="", plot_kind="3D", dim=None, fig=None, plot_position=None, show=True):
        """
        Plot a figure of the ratings on a 3D or 2D space
        with the voters having the `sizes` passed
        as parameters.

        Parameters
        ----------
        sizes : np.ndarray
            The score given by each voter.
            Should be of length :attr:`n_voters`.
        title : str
            Title of the figure.
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        fig : matplotlib figure
            The figure on which we add the plot.
        plot_position : list
            The position of the plot on the figure.
            Should be of the form
            ``[n_rows, n_columns, position]``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        show : bool
            If True, display the figure at
            the end of the function.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure,
            if you want to add something to it.

        """
        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        if plot_kind == "3D":
            ax = self._plot_scores_3D(sizes, fig, plot_position, dim)
        elif plot_kind == "ternary":
            ax = self._plot_scores_ternary(sizes, fig, plot_position, dim)
        else:
            raise ValueError("plot_kind should '3D' or 'ternary'")

        ax.set_title(title, fontsize=16)
        if show:
            plt.show()  # pragma: no cover

        return ax

    def plot_candidate(self, ratings, candidate, plot_kind="3D", dim=None, fig=None, plot_position=None, show=True):
        """
        Plot a figure of the ratings
        with the voters having the ratings they give
        to the `candidate` passed as parameters
        as size.

        Parameters
        ----------
        ratings: np.ndarray
            Matrix of ratings given by voters to candidates
        candidate : int
            The candidate for which we
            want to show the ratings.
            Should be lower than :attr:`n_candidates`.
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        fig : matplotlib figure
            The figure on which we add the plot.
        plot_position : list
            The position of the plot on the figure.
            Should be of the form
            ``[n_rows, n_columns, position]``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        show : bool
            If True, display the figure
            at the end of the function.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure,
            if you want to add something to it.
        """
        return self.plot_scores(ratings[::, candidate],
                                title="Candidate %i" % (candidate + 1),
                                plot_kind=plot_kind,
                                dim=dim,
                                fig=fig,
                                plot_position=plot_position,
                                show=show)

    def plot_candidates(self, ratings, plot_kind="3D", dim=None, list_candidates=None,
                        list_titles=None, row_size=5, show=True):
        """
        Plot the ratings of the voters
        for every candidate or a list of candidates,
        using the ratings given by the voters as size for
        the voters. The plot is either on a 3D plot,
        or on a ternary plot.
        Only three dimensions can be represented.

        Parameters
        ----------
        ratings: Ratings
            Ratings given by voters to candidates
        plot_kind : str
            The kind of plot we want to show.
            Can be ``'3D'`` or ``'ternary'``.
        dim : list
            The 3 dimensions we are using for our plot.
            By default, it is set to ``[0, 1, 2]``.
        list_candidates : int list
            The list of candidates we want to plot.
            Should contains integer lower than
            :attr:`n_candidates`.
            By default, we plot every candidates.
        list_titles : str list
            Contains the title of the plots.
            Should be the same length than `list_candidates`.
        row_size : int
            Number of subplots by row.
            By default, it is set to 5 plots by rows.
        show : bool
            If True, display the figure
            at the end of the function.

        """
        if not isinstance(ratings, np.ndarray):
            ratings = ratings.ratings_
        if list_candidates is None:
            list_candidates = range(ratings.shape[1])
        if list_titles is None:
            list_titles = ["Candidate %i" % c for c in list_candidates]
        else:
            list_titles = ["%s " % t for t in list_titles]

        n_candidates = len(list_candidates)
        n_rows = (n_candidates - 1) // row_size + 1
        fig = plt.figure(figsize=(5 * row_size, n_rows * 5))
        position = [n_rows, row_size, 1]
        for candidate, title in (zip(list_candidates, list_titles)):
            self.plot_scores(ratings[::, candidate],
                             title=title,
                             plot_kind=plot_kind,
                             dim=dim,
                             fig=fig,
                             plot_position=position,
                             show=False)
            position[2] += 1

        if show:
            plt.show()  # pragma: no cover

    def copy(self):
        """
        Return a copy of the embeddings.

        Return
        ------
        Embeddings
            A copy of the embeddings.

        Examples
        --------
        >>> embs = Embeddings(np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]]), norm=False)
        >>> second_embs = embs.copy()
        >>> second_embs
        Embeddings([[0.5, 0.9, 0.4],
                    [0.4, 0.7, 0.5],
                    [0.4, 0.2, 0.4]])
        """
        return Embeddings(self, norm=False)
