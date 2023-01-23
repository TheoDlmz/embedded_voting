# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.utils.miscellaneous import normalize, max_angular_dilatation_factor, volume_parallelepiped
from embedded_voting.utils.plots import create_ternary_plot, create_3d_plot


# noinspection PyUnresolvedReferences
class Embeddings(np.ndarray):
    """
    Embeddings of the voters.

    Parameters
    ----------
    positions : np.ndarray or list or Embeddings
        The embeddings of the voters. Its dimensions are :attr:`n_voters`, :attr:`n_dim`.
    norm: bool
        If True, normalize the embeddings.

    Attributes
    ----------
    n_voters : int
        The number of voters in the ratings.
    n_dim : int
        The number of dimensions of the voters' embeddings.

    Examples
    --------
    >>> embeddings = Embeddings([[1, 0], [0, 1], [0.5, 0.5]], norm=True)
    >>> embeddings.n_voters
    3
    >>> embeddings.n_dim
    2
    >>> embeddings.voter_embeddings(0)
    array([1., 0.])
    """

    def __new__(cls, positions, norm):
        """
        >>> embeddings = Embeddings([[1, 2], [3, 4]], norm=False)
        >>> embeddings.n_sing_val_ = 42
        >>> embeddings_2 = Embeddings(embeddings, norm=False)
        >>> embeddings_2.n_sing_val_
        42
        """
        obj = np.asarray(positions).view(cls)
        if norm:
            obj = obj / np.linalg.norm(obj, axis=1)[:, np.newaxis]
        if hasattr(positions, '__dict__'):
            for key, val in positions.__dict__.items():
                setattr(obj, key, getattr(positions, key))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if len(self.shape) == 2:
            self.n_voters, self.n_dim = self.shape

    def copy(self, order='C'):
        result = super().copy(order=order)
        for key, val in self.__dict__.items():
            setattr(result, key, getattr(self, key))
        return result

    def voter_embeddings(self, i):
        return np.array(self[i:i + 1, :])[0]

    def times_ratings_candidate(self, ratings_candidate):
        """
        This method computes the embeddings multiplied by the ratings given by the voters to a given candidate.
        For each voter, its embeddings are multiplied by the given rating.

        Parameters
        ----------
        ratings_candidate: np.ndarray
            The vector of ratings given by the voters to a given candidate.

        Return
        ------
        Embddings
            A new Embeddings object, where the embedding of each voter is multiplied by the rating she assigned
            to the candidate.

        Examples
        --------
        >>> embeddings = Embeddings(np.array([[1, 0], [0, 1], [0.5, 0.5]]), norm=False)
        >>> embeddings.times_ratings_candidate(np.array([.8, .5, .4]))
        Embeddings([[0.8, 0. ],
                    [0. , 0.5],
                    [0.2, 0.2]])
        """
        return np.multiply(self, ratings_candidate[::, np.newaxis])

    def normalized(self):
        """
        Normalize the embeddings of the voters so the Euclidean norm of every embedding is 1.

        Return
        ------
        Embeddings
            A new Embeddings object with the normalized embeddings.

        Examples
        --------
        >>> embeddings = Embeddings(-np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]]), norm=False)
        >>> embeddings
        Embeddings([[-0.5, -0.9, -0.4],
                    [-0.4, -0.7, -0.5],
                    [-0.4, -0.2, -0.4]])
        >>> embeddings.normalized()
        Embeddings([[-0.45267873, -0.81482171, -0.36214298],
                    [-0.42163702, -0.73786479, -0.52704628],
                    [-0.66666667, -0.33333333, -0.66666667]])
        """
        return self / np.linalg.norm(self, axis=1)[:, np.newaxis]

    def get_center(self, approx=True):
        """
        Return the center direction of the embeddings.

        For this method, we work on the normalized embeddings. Cf. :meth:`normalized`.

        With `approx` set to False, we use an exponential algorithm in `n_dim.`
        If `r` is the rank of the embedding matrix, we first find the `r` voters with
        maximal determinant (in absolute value), i.e. whose associated parallelepiped has
        the maximal volume (e.g. in two dimensions, it means finding the two vectors with
        maximal angle). Then the result is the mean of the embeddings of these voters,
        normalized in the sense of the Euclidean norm.

        With `approx` set to True, we use a polynomial algorithm: we simply take the mean
        of the embeddings of all the voters, normalized in the sense of the Euclidean norm.

        Parameters
        ----------
        approx : bool
            Whether the computation is approximate.

        Return
        ------
        np.ndarray
            The normalized position of the center vector. Size: :attr:`n_dim`.

        Examples
        --------
        >>> embeddings = Embeddings([[1, 0], [0, 1], [.5, .5], [.7, .3]], norm=True)
        >>> embeddings.get_center(approx=False)
        array([0.70710678, 0.70710678])

        >>> embeddings = Embeddings([[1, 0], [0, 1], [.5, .5], [.7, .3]], norm=False)
        >>> embeddings.get_center(approx=False)
        array([0.70710678, 0.70710678])

        >>> embeddings = Embeddings([[1, 0], [0, 1], [.5, .5], [.7, .3]], norm=True)
        >>> embeddings.get_center(approx=True)
        array([0.78086524, 0.62469951])
        """
        self_normalized = self.normalized()
        if approx:
            return normalize(np.array(self_normalized.sum(axis=0)))
        else:
            matrix_rank = np.linalg.matrix_rank(self_normalized)
            subset_of_voters = max(
                combinations(range(self.n_voters), matrix_rank),
                key=lambda subset: volume_parallelepiped(self_normalized[subset, :])
            )
            embeddings_subset = self_normalized[subset_of_voters, :]
            mean = normalize(np.array(embeddings_subset.sum(axis=0)))
            return mean

    def dilated_aux(self, center, k):
        """
        Dilate the embeddings of the voters.

        For each `vector` of the embedding, we apply a "spherical dilatation" that moves `vector`
        by multiplying the angle between `center` and `vector` by a given dilatation factor.

        More formally, for each `vector` of the embedding, there exists a unit vector `unit_orthogonal` and
        an angle `theta in [0, pi/2]` such that
        `vector = norm(vector) * (cos(theta) * center + sin(theta) * unit_orthogonal)`.
        Then the image of `vector` is
        `norm(vector) * (cos(k * theta) * center + sin(k * theta) * unit_orthogonal)`.

        Parameters
        ----------
        center: np.ndarray
            Unit vector: center of the dilatation.
        k: float
            Angular dilatation factor.

        Return
        ------
        Embeddings
            A new Embeddings object with the dilated embeddings.

        Examples
        --------
        >>> embeddings = Embeddings([[1, 0], [1, 1]], norm=True)
        >>> dilated_embeddings = embeddings.dilated_aux(center=np.array([1, 0]), k=2)
        >>> np.round(dilated_embeddings, 4)
        array([[1., 0.],
               [0., 1.]])

        >>> embeddings = Embeddings([[1, 0], [1, 1]], norm=False)
        >>> dilated_embeddings = embeddings.dilated_aux(center=np.array([1, 0]), k=2)
        >>> np.abs(np.round(dilated_embeddings, 4)) # Abs for rounding errors
        array([[1.    , 0.    ],
               [0.    , 1.4142]])
        """
        new_positions = np.zeros((self.n_voters, self.n_dim))
        for i in range(self.n_voters):
            vector = self.voter_embeddings(i)
            norm_vector = np.linalg.norm(vector)
            scalar_product = vector @ center
            theta = np.arccos(scalar_product / norm_vector)
            if theta == 0:
                new_positions[i] = vector
            else:
                vector_collinear = scalar_product * center
                vector_orthogonal = vector - vector_collinear
                unit_orthogonal = normalize(vector_orthogonal)
                new_positions[i] = norm_vector * (center * np.cos(k * theta) + unit_orthogonal * np.sin(k * theta))
        return Embeddings(new_positions, norm=False)

    def dilated(self, approx=True):
        """
        Dilate the embeddings of the voters so that they take more space.

        The `center` is computed with :meth:`get_center`. The angular dilatation
        factor is such that after transformation, the maximum angle between
        the center and an embedding vector will be pi / 4.

        Parameters
        ----------
        approx : bool
            Passed to :meth:`get_center` in order to compute the center
            of the voters' embeddings.

        Return
        ------
        Embeddings
            A new Embeddings object with the dilated embeddings.

        Examples
        --------
        >>> embeddings = Embeddings(np.array([[.5,.4,.4],[.4,.4,.5],[.4,.5,.4]]), norm=True)
        >>> embeddings
        Embeddings([[0.66226618, 0.52981294, 0.52981294],
                    [0.52981294, 0.52981294, 0.66226618],
                    [0.52981294, 0.66226618, 0.52981294]])
        >>> embeddings.dilated()
        Embeddings([[0.98559856, 0.11957316, 0.11957316],
                    [0.11957316, 0.11957316, 0.98559856],
                    [0.11957316, 0.98559856, 0.11957316]])

        Note that the resulting embedding may not be in the positive orthant,
        even if the original embedding is:

        >>> embeddings = Embeddings([[1, 0], [.7, .7]], norm=True)
        >>> embeddings.dilated()
        Embeddings([[ 0.92387953, -0.38268343],
                    [ 0.38268343,  0.92387953]])

        >>> Embeddings([[1, 0]], norm=True).dilated()
        Embeddings([[1., 0.]])
        """
        center = self.get_center(approx=approx)
        min_scalar_product = min([
            np.dot(normalize(self.voter_embeddings(i)), center)
            for i in range(self.n_voters)
        ])
        theta_max = np.arccos(min_scalar_product)
        if theta_max == 0:  # all embeddings are aligned with `center`.
            return self.copy()
        k = np.pi / (4 * theta_max)
        return self.dilated_aux(center=center, k=k)

    def dilated_new(self, approx=True):
        """
        Dilate the embeddings of the voters so that they take more space in the positive orthant.

        The `center` is computed with :meth:`get_center`. The angular dilatation
        factor the largest possible so that all vectors stay in the positive orthant.
        Cf. :func:`~embedded_voting.utils.miscellaneous.max_angular_dilatation_factor`.

        Parameters
        ----------
        approx : bool
            Passed to :meth:`get_center` in order to compute the center
            of the voters' embeddings.

        Return
        ------
        Embeddings
            A new Embeddings object with the dilated embeddings.

        Examples
        --------
        >>> embeddings = Embeddings(np.array([[.5,.4,.4],[.4,.4,.5],[.4,.5,.4]]), norm=True)
        >>> embeddings
        Embeddings([[0.66226618, 0.52981294, 0.52981294],
                    [0.52981294, 0.52981294, 0.66226618],
                    [0.52981294, 0.66226618, 0.52981294]])
        >>> dilated_embeddings = embeddings.dilated_new()
        >>> np.abs(np.round(dilated_embeddings, 4))
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])

        >>> embeddings = Embeddings([[1, 0], [.7, .7]], norm=True)
        >>> dilated_embeddings = embeddings.dilated_new()
        >>> np.abs(np.round(dilated_embeddings, 4))
        array([[1.    , 0.    ],
               [0.7071, 0.7071]])

        >>> embeddings = Embeddings([[2, 1], [100, 200]], norm=False)
        >>> dilated_embeddings = embeddings.dilated_new()
        >>> np.round(dilated_embeddings, 4)
        array([[  2.2361,   0.    ],
               [  0.    , 223.6068]])
        """
        center = self.get_center(approx=approx)
        k = min([
            max_angular_dilatation_factor(vector=np.array(vector), center=center)
            for vector in self.normalized()
        ])
        if np.isinf(k):
            return self.copy()
        return self.dilated_aux(center=center, k=k)

    def recentered(self, approx=True):
        """
        Recenter the embeddings so that their new center is [1, ..., 1].

        Parameters
        ----------
        approx : bool
            Passed to :meth:`get_center` in order to compute the center
            of the voters' embeddings.

        Return
        ------
        Embeddings
            A new Embeddings object with the recentered embeddings.

        Examples
        --------
        >>> embeddings = Embeddings(-np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]]), norm=True)
        >>> embeddings
        Embeddings([[-0.45267873, -0.81482171, -0.36214298],
                    [-0.42163702, -0.73786479, -0.52704628],
                    [-0.66666667, -0.33333333, -0.66666667]])
        >>> embeddings.recentered()
        Embeddings([[0.40215359, 0.75125134, 0.52334875],
                    [0.56352875, 0.6747875 , 0.47654713],
                    [0.70288844, 0.24253193, 0.66867489]])

        >>> embeddings = Embeddings([[1, 0], [np.sqrt(3)/2, 1/2], [1/2, np.sqrt(3)/2]], norm=True)
        >>> embeddings
        Embeddings([[1.       , 0.       ],
                    [0.8660254, 0.5      ],
                    [0.5      , 0.8660254]])
        >>> embeddings.recentered(approx=False)
        Embeddings([[0.96592583, 0.25881905],
                    [0.70710678, 0.70710678],
                    [0.25881905, 0.96592583]])
        """
        old_center = self.get_center(approx=approx)
        target_center = normalize(np.ones(self.n_dim))
        scalar_product = old_center @ target_center
        if scalar_product == -1:
            return - self
        elif scalar_product == 1:
            return self.copy()
        theta = np.arccos(scalar_product)
        target_center_collinear = scalar_product * old_center
        target_center_orthogonal = target_center - target_center_collinear
        unit_orthogonal = target_center_orthogonal / np.linalg.norm(target_center_orthogonal)
        # (old_center, unit_orthogonal) is an orthogonal basis of the plane (old_center, target_center).
        rotation_matrix = (
            (np.cos(theta) - 1) * (
                np.outer(old_center, old_center)
                + np.outer(unit_orthogonal, unit_orthogonal)
            )
            + np.sin(theta) * (
                np.outer(unit_orthogonal, old_center)
                - np.outer(old_center, unit_orthogonal)
            )
            + np.eye(self.n_dim)
        )
        new_positions = self @ rotation_matrix.T
        return Embeddings(new_positions, norm=False)

    def recentered_and_dilated(self, approx=True):
        """
        Recenter and dilate.

        This is just a shortcut for the (common) operation :meth:`recentered`, then :meth:`dilated_new`.

        Parameters
        ----------
        approx : bool
            Passed to :meth:`get_center` in order to compute the center
            of the voters' embeddings.

        Returns
        -------
        Embeddings
            A new Embeddings object with the recentered and dilated embeddings.

        Examples
        --------
        >>> embeddings = Embeddings([[1, 0], [np.sqrt(3)/2, 1/2], [1/2, np.sqrt(3)/2]], norm=True)
        >>> embeddings
        Embeddings([[1.       , 0.       ],
                    [0.8660254, 0.5      ],
                    [0.5      , 0.8660254]])
        >>> new_embeddings = embeddings.recentered_and_dilated(approx=False)
        >>> np.abs(np.round(new_embeddings, 4))
        array([[1.    , 0.    ],
               [0.7071, 0.7071],
               [0.    , 1.    ]])
        """
        return self.recentered(approx=approx).dilated_new(approx=approx)

    def mixed_with(self, other, intensity):
        """
        Mix this embedding with another one.

        Parameters
        ----------
        other : Embeddings
            Another embedding with the name number of voters and same number of dimensions.
        intensity : float
            Must be in [0, 1].

        Returns
        -------
        Embeddings
            A new Embeddings object with the mixed embeddings.

        Examples
        --------
        For a given voter, the direction of the final embedding is an "angular barycenter" between
        the original direction and the direction in `other`, with mixing parameter `intensity`:

        >>> embeddings = Embeddings([[1, 0]], norm=True)
        >>> other_embeddings = Embeddings([[0, 1]], norm=True)
        >>> embeddings.mixed_with(other_embeddings, intensity=1/3)
        Embeddings([[0.8660254, 0.5      ]])

        For a given voter, the norm of the final embedding is a barycenter between the original
        norm and the norm in `other`, with mixing parameter `intensity`:

        >>> embeddings = Embeddings([[1, 0]], norm=False)
        >>> other_embeddings = Embeddings([[5, 0]], norm=False)
        >>> embeddings.mixed_with(other_embeddings, intensity=1/4)
        Embeddings([[2., 0.]])
        """
        norms_self = np.linalg.norm(self, axis=1)
        norms_other = np.linalg.norm(other, axis=1)
        self_normalized = self / norms_self[:, np.newaxis]
        other_dot_products_self_normalized = np.sum(other * self_normalized, axis=1)
        other_collinear = other_dot_products_self_normalized[:, np.newaxis] * self_normalized
        other_orthogonal = other - other_collinear
        norms_other_orthogonal = np.linalg.norm(other_orthogonal, axis=1)
        unit_orthogonal = other_orthogonal / np.where(
            norms_other_orthogonal > 0, norms_other_orthogonal, 1
        )[:, np.newaxis]
        thetas = np.arccos(other_dot_products_self_normalized / norms_other)
        norms = (1 - intensity) * norms_self + intensity * norms_other
        directions = (
            np.cos(intensity * thetas)[:, np.newaxis] * self_normalized
            + np.sin(intensity * thetas)[:, np.newaxis] * unit_orthogonal
        )
        return norms[:, np.newaxis] * directions

    def _plot_3d(self, fig, dim, plot_position=None):
        """
        Plot a figure of the embeddings on a 3D space using matplotlib.

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
            The matplotlib ax with the figure, if
            you want to add something to it.
        """
        ax = create_3d_plot(fig, plot_position)
        for v in self:
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            ax.plot([0, x1], [0, x2], [0, x3], color=(x1**2 * 0.8, x2**2 * 0.8, x3**2 * 0.8), alpha=0.4)
            ax.scatter([x1], [x2], [x3], color='k', s=1)
        return ax

    def _plot_ternary(self, fig, dim, plot_position=None):
        """
        Plot a figure of the embeddings on a 2D space
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
        for v in self:
            x1 = v[dim[0]]
            x2 = v[dim[2]]
            x3 = v[dim[1]]
            vec = [x1, x2, x3]
            tax.scatter([normalize(vec)**2], color=(x1**2 * 0.8, x3**2 * 0.8, x2**2 * 0.8), alpha=0.9, s=30)
        return tax

    def plot(self, plot_kind="3D", dim: list = None, fig=None, plot_position=None, show=True):
        """
        Plot the embeddings of the voters,
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
            `8 x 8` matplotlib figure.
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
        elif len(dim) != 3:
            raise ValueError("The number of dimensions should be 3")
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        if plot_kind == "3D":
            ax = self._plot_3d(fig, dim, plot_position)
        elif plot_kind == "ternary":
            ax = self._plot_ternary(fig, dim, plot_position)
        else:
            raise ValueError("plot_kind should '3D' or 'ternary'")
        ax.set_title("Embeddings of voters on dimensions (%i,%i,%i)" % (dim[0], dim[1], dim[2]), fontsize=24)
        if show:
            plt.show()  # pragma: no cover
        return ax

    def _plot_ratings_candidate_3d(self, ratings_candidate, fig, plot_position, dim):
        """
        Plot the matrix associated to a candidate in a 3D space.

        The embedding of each voter is multiplied by the rating she assigned to the candidate.

        Parameters
        ----------
        ratings_candidate : np.ndarray
            The rating each voters assigned to the given candidate.
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
        ax = create_3d_plot(fig, plot_position)
        for (v, s) in zip(np.array(self), ratings_candidate):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            ax.plot([0, s * x1], [0, s * x2], [0, s * x3], color=(x1**2 * 0.8, x2**2 * 0.8, x3**2 * 0.8), alpha=0.4)
        return ax

    def _plot_ratings_candidate_ternary(self, ratings_candidate, fig, plot_position, dim):
        """
        Plot the matrix associated to a candidate on a 2D space representing the sphere in the non-negative orthant.

        The embedding of each voter is multiplied by the rating she assigned to the candidate.

        Parameters
        ----------
        ratings_candidate : np.ndarray
            The rating each voters assigned to the given candidate.
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
        for (v, s) in zip(np.array(self), ratings_candidate):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            vec = [x1, x2, x3]
            tax.scatter([normalize(vec)**2], color=(x1**2 * 0.8, x3**2 * 0.8, x2**2 * 0.8), alpha=0.7, s=max(s * 50, 1))
        return tax

    def plot_ratings_candidate(self, ratings_candidate, title="", plot_kind="3D", dim: list = None, fig=None,
                               plot_position=None, show=True):
        """
        Plot the matrix associated to a candidate.

        The embedding of each voter is multiplied by the rating she assigned to the candidate.

        Parameters
        ----------
        ratings_candidate : np.ndarray
            The rating each voters assigned to the given candidate.
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
        elif len(dim) != 3:
            raise ValueError("The number of dimensions should be 3")
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        if plot_kind == "3D":
            ax = self._plot_ratings_candidate_3d(ratings_candidate, fig, plot_position, dim)
        elif plot_kind == "ternary":
            ax = self._plot_ratings_candidate_ternary(ratings_candidate, fig, plot_position, dim)
        else:
            raise ValueError("plot_kind should '3D' or 'ternary'")
        ax.set_title(title, fontsize=16)
        if show:
            plt.show()  # pragma: no cover
        return ax

    def plot_candidate(self, ratings, candidate, plot_kind="3D", dim: list = None, fig=None, plot_position=None,
                       show=True):
        """
        Plot the matrix associated to a candidate.

        The embedding of each voter is multiplied by the rating she assigned to the candidate.

        Parameters
        ----------
        ratings: np.ndarray
            Matrix of ratings given by voters to the candidates.
        candidate : int
            The candidate for which we want to show the ratings.
            Should be lower than :attr:`n_candidates` of ratings.
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
        return self.plot_ratings_candidate(ratings[::, candidate],
                                           title="Candidate %i" % (candidate + 1),
                                           plot_kind=plot_kind,
                                           dim=dim,
                                           fig=fig,
                                           plot_position=plot_position,
                                           show=show)

    def plot_candidates(self, ratings, plot_kind="3D", dim: list = None, list_candidates=None,
                        list_titles=None, row_size=5, show=True):
        """
        Plot the matrix associated to a candidate for every candidate in a list of candidates.

        Parameters
        ----------
        ratings: Ratings
            Ratings given by voters to candidates.
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
            self.plot_ratings_candidate(ratings[::, candidate],
                                        title=title,
                                        plot_kind=plot_kind,
                                        dim=dim,
                                        fig=fig,
                                        plot_position=position,
                                        show=False)
            position[2] += 1
        if show:
            plt.show()  # pragma: no cover
