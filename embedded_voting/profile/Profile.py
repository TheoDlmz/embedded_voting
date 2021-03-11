# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.utils.miscellaneous import normalize
from embedded_voting.utils.cached import DeleteCacheMixin
from embedded_voting.utils.plots import create_ternary_plot, create_3D_plot


class Profile(DeleteCacheMixin):
    """
    A profile of voter

    Parameters
    ----------
    n_candidates : int
        The number of candidate for this profile
    n_dim : int
        The number of dimensions for the voters' embeddings

    Attributes
    ----------
    n_voters : int
        The number of voter of the profile
    n_candidates : int
        The number of candidate for this profile
    n_dim : int
        The number of dimensions for the voters' embeddings
    embeddings :np.ndarray
        The embeddings of the voters. Its dimensions are :attr:`n_voters`, :attr:`n_dim`
    scores : np.ndarray
        The scores given by the voters to the candidates.
        Its dimensions are :attr:`n_voters`, :attr:`n_candidates`

    Examples
    --------
    >>> my_profile = Profile(5, 3)
    >>> my_profile.uniform_distribution(100)
    <embedded_voting.profile.Profile.Profile object at ...>
    >>> my_profile.add_voter([.1, .5, .5], [1]*5)
    <embedded_voting.profile.Profile.Profile object at ...>
    >>> my_profile.n_candidates
    5
    >>> my_profile.n_dim
    3
    >>> my_profile.n_voters
    101
    >>> my_profile.embeddings[-1]
    array([0.14002801, 0.70014004, 0.70014004])
    >>> my_profile.scores[-1]
    array([1., 1., 1., 1., 1.])
    """

    def __init__(self, n_candidates, n_dim):
        self.embeddings = np.zeros((0, n_dim))
        self.scores = np.zeros((0, n_candidates))
        self.n_candidates = n_candidates
        self.n_dim = n_dim
        self.n_voters = 0

    def add_voter(self, embeddings, scores, normalize_embs=True):
        """
        Add one voter to the profile.

        Parameters
        ----------
        embeddings : np.ndarray or list
            The embedding vector of the voter.
            Should be of size :attr:`n_dim`
        scores : np.ndarray or list
            The scores given by the voter to the candidates.
            Should be of size :attr:`n_candidates`
        normalize_embs : bool
            If True, then normalize the embeddings in parameter.

        Return
        ------
        Profile
            The profile itself

        Examples
        ________
        >>> my_profile = Profile(5, 3)
        >>> my_profile.n_voters
        0
        >>> my_profile.add_voter([.1, .5, .5], [1]*5)
        <embedded_voting.profile.Profile.Profile object at ...>
        >>> my_profile.n_voters
        1
        """
        embeddings = np.array(embeddings)
        if normalize_embs:
            embeddings = normalize(embeddings)
        self.embeddings = np.concatenate([self.embeddings, [embeddings]])
        self.scores = np.concatenate([self.scores, [scores]])
        self.n_voters += 1

        return self

    def add_voters(self, embeddings, scores, normalize_embs=True):
        """
        Add a group of n voters to the profile

        Parameters
        ----------
        embeddings : np.ndarray or list
            The embeddings vectors of the new voters.
            Should be of size `n_new_voters`, :attr:`n_dim`
        scores : np.ndarray or list
            The scores given by the new voters.
            Should be of size `n_new_voters`, :attr:`n_candidates`
        normalize_embs : bool
            If True, then normalize the embeddings in parameter.

        Return
        ------
        Profile
            The profile itself

        Examples
        ________
        >>> my_profile = Profile(5, 3)
        >>> my_profile.n_voters
        0
        >>> my_profile.add_voters(np.random.rand(10, 3), np.random.rand(10, 5))
        <embedded_voting.profile.Profile.Profile object at ...>
        >>> my_profile.n_voters
        10
        """
        embeddings = np.array(embeddings)
        if normalize_embs:
            embeddings = (embeddings.T / np.sqrt((embeddings ** 2).sum(axis=1))).T
        self.embeddings = np.concatenate([self.embeddings, embeddings])
        self.scores = np.concatenate([self.scores, scores])
        self.n_voters += len(embeddings)

        return self

    def uniform_distribution(self, n_voters):
        """
        Add `n_voters` voters to the profile.
        The embeddings of these voters are uniformly distributed on the positive ortan and there score
        are uniformly distributed between 0 and 1.

        Parameters
        ----------
        n_voters : int
            Number of new voters to add in the profile

        Return
        ------
        Profile
            The profile itself

        Examples
        ________
        >>> my_profile = Profile(5, 3)
        >>> my_profile.n_voters
        0
        >>> my_profile.uniform_distribution(100)
        <embedded_voting.profile.Profile.Profile object at ...>
        >>> my_profile.n_voters
        100
        """

        new_group = np.zeros((n_voters, self.n_dim))

        for i in range(n_voters):
            new_vec = np.abs(np.random.randn(self.n_dim))
            new_vec = normalize(new_vec)
            new_group[i] = new_vec

        self.embeddings = np.concatenate([self.embeddings, new_group])
        self.n_voters += n_voters

        new_scores = np.random.rand(n_voters, self.n_candidates)
        self.scores = np.concatenate([self.scores, new_scores])

        return self

    def dilate_profile(self):
        """
        Dilate the embeddings of the voters so that the
        embeddings take all the space possible in the positive ortan.

        Return
        ------
        Profile
            The profile itself

        Examples
        --------
        >>> my_profile = Profile(5, 3)
        >>> embeddings = np.array([[.5,.4,.4],[.4,.4,.5],[.4,.5,.4]])
        >>> scores = np.random.rand(3, 5)
        >>> _ = my_profile.add_voters(embeddings, scores, normalize_embs=True)
        >>> my_profile.embeddings
        array([[0.66226618, 0.52981294, 0.52981294],
               [0.52981294, 0.52981294, 0.66226618],
               [0.52981294, 0.66226618, 0.52981294]])
        >>> my_profile.dilate_profile().embeddings
        array([[0.98559856, 0.11957316, 0.11957316],
               [0.11957316, 0.11957316, 0.98559856],
               [0.11957316, 0.98559856, 0.11957316]])
        """

        profile = self.embeddings

        if self.n_voters < 2:
            raise ValueError("Cannot dilate a profile with less than 2 candidates")

        center = self.embeddings.sum(axis=0)
        center = center / np.linalg.norm(center)
        min_value = np.dot(profile[0], center)
        for i in range(self.n_voters):
            val = np.dot(profile[i], center)
            if val < min_value:
                min_value = val

        theta_max = np.arccos(min_value)
        k = np.pi / (4 * theta_max)

        new_profile = np.zeros((self.n_voters, self.n_dim))
        for i in range(self.n_voters):
            v_i = self.embeddings[i]
            theta_i = np.arccos(np.dot(v_i, center))
            if theta_i == 0:
                new_profile[i] = v_i
            else:
                p_1 = np.dot(center, v_i) * center
                p_2 = v_i - p_1
                e_2 = normalize(p_2)
                new_profile[i] = center * np.cos(k * theta_i) + e_2 * np.sin(k * theta_i)

        self.embeddings = new_profile

        return self

    def standardize(self, cut_one=True):
        """
        Standardize the score between the different voters.

        Parameters
        ----------
        cut_one : bool
            if True, then the maximum score is one. The minimum score is always 0

        Return
        ------
        Profile
            The profile itself

        Examples
        --------
        >>> my_profile = Profile(3, 3)
        >>> _ = my_profile.add_voter(np.random.rand(3), [.6, .8, 1])
        >>> _ = my_profile.add_voter(np.random.rand(3), [.4, .5, .3])
        >>> my_profile.scores.mean(axis=0)
        array([0.5 , 0.65, 0.65])
        >>> my_profile.standardize().scores.mean(axis=0)
        array([0.25, 0.75, 0.5 ])
        """

        mu = self.scores.mean(axis=1)
        sigma = self.scores.std(axis=1)
        new_scores = self.scores.T - np.tile(mu, (self.n_candidates, 1))
        new_scores = new_scores / sigma
        new_scores += 1
        new_scores /= 2
        new_scores = np.maximum(new_scores, 0)
        if cut_one:
            new_scores = np.minimum(new_scores, 1)
        self.scores = new_scores.T

        return self

    def scored_embeddings(self, candidate, square_root=True):
        """
        Return the embeddings matrix with each voter's embedding multiplied by the score it gives
        to the given candidate.

        Parameters
        ----------
        candidate : int
            the candidate of who we use the scores.
        square_root : bool
            if True, we multiply by the square root of the score given to the candidate
            instead of the score itself.

        Return
        ------
        np.ndarray
            The embedding matrix, of shape :attr:`n_voters`, :attr:`n_dim`

        Examples
        --------
        >>> my_profile = Profile(1, 3)
        >>> _ = my_profile.add_voter([0, .5, .5], [.5])
        >>> _ = my_profile.add_voter([.5, .5, 0], [.2])
        >>> my_profile.embeddings
        array([[0.        , 0.70710678, 0.70710678],
               [0.70710678, 0.70710678, 0.        ]])
        >>> my_profile.scored_embeddings(0)
        array([[0.        , 0.5       , 0.5       ],
               [0.31622777, 0.31622777, 0.        ]])
        >>> my_profile.scored_embeddings(0, square_root=False)
        array([[0.        , 0.35355339, 0.35355339],
               [0.14142136, 0.14142136, 0.        ]])
        """

        embeddings = []
        for i in range(self.n_voters):
            if square_root:
                s = np.sqrt(self.scores[i, candidate])
            else:
                s = self.scores[i, candidate]
            embeddings.append(self.embeddings[i] * s)
        return np.array(embeddings)

    def fake_covariance_matrix(self, candidate, f, square_root=True):
        """
        This function return a matrix `M` such that for all voters `i`  and `j`,
        `M[i,j] = scores[i, candidate] * scores[j, candidate]
        * f(embeddings[i], embeddings[j])` (cf :attr:`scores`, :attr:`embeddings`)

        Parameters
        ----------
        candidate : int
            Candidate for which we want the matrix
        f : callable
            Similarity function between two embeddings vector of length :attr:`n_dim`.
            Input : (np.ndarray, np.ndarray).
            Output : float.
        square_root : bool
            If True, use the square root of the score instead of the scores itself

        Return
        ------
        np.ndarray
            Matrix of shape :attr:`n_voters`, :attr:`n_voters`

        Examples
        --------
        >>> my_profile = Profile(1, 3)
        >>> _ = my_profile.add_voter([0, .5, .5], [.5])
        >>> _ = my_profile.add_voter([.5, .5, 0], [.2])
        >>> my_profile.fake_covariance_matrix(0, np.dot)
        array([[0.5       , 0.15811388],
               [0.15811388, 0.2       ]])
        """
        matrix = np.zeros((self.n_voters, self.n_voters))

        for i in range(self.n_voters):
            for j in range(i, self.n_voters):
                s = self.scores[i, candidate] * self.scores[j, candidate]
                if square_root:
                    s = np.sqrt(s)
                matrix[i, j] = f(self.embeddings[i], self.embeddings[j]) * s
                matrix[j, i] = matrix[i, j]

        return matrix

    def _plot_profile_3D(self, fig, dim, position=None):
        """
        Plot a figure of the profile on a 3D space

        Parameters
        ----------
        fig : matplotlib.figure
            The figure on which we add the plot.
        dim : list
            The 3 dimensions we are using for our plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.

        Return
        ------
        matplotlib.ax
            The matplotlib ax with the figure, if you want to add something to it.

        """
        ax = create_3D_plot(fig, position)
        for i, v in enumerate(self.embeddings):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            ax.plot([0, x1], [0, x2], [0, x3], color=(x1 * 0.8, x2 * 0.8, x3 * 0.8), alpha=0.4)
            ax.scatter([x1], [x2], [x3], color='k', s=1)
        return ax

    def _plot_profile_ternary(self, fig, dim, position=None):
        """
        Plot a figure of the profile on a 2D space representing the surface of the unit sphere
        on the positive ortan.

        Parameters
        ----------
        fig : matplotlib figure
            The figure on which we add the plot.
        dim : list
            The 3 dimensions we are using for our plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure, if you want to add something to it.

        """
        tax = create_ternary_plot(fig, position)
        for i, v in enumerate(self.embeddings):
            x1 = v[dim[0]]
            x2 = v[dim[2]]
            x3 = v[dim[1]]
            vec = [x1, x2, x3]
            tax.scatter([normalize(vec)**2], color=(x1 * 0.8, x3 * 0.8, x2 * 0.8), alpha=0.9, s=30)

        return tax

    def plot_profile(self, plot_kind="3D", dim=None, fig=None, position=None, show=True):
        """
        Plot the profile of the voters, either on a 3D plot, or on a ternary plot. Only
        three dimensions are represented.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        dim : list
            A list of length 3 containing the three dimensions of the embeddings we want to plot.
            All elements of this list should be lower than :attr:`n_dim`. By default, it is
            set are `[0, 1, 2]`.
        fig : matplotlib figure
            The figure on which we add the plot.
            if None, the default figure is a `10 x 10` matplotlib figure.
        position : list
            List of length 3 containing the matplotlib position `[a, b, c]` of the figure, where
            `a` is the number of rows, `b` is the number of columns and `c` is the position of the figure
            (from top to bottom, left to right). By default, it is set to `[1, 1, 1]`
        show : bool
            if True, show the figure at the end of the function.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure, if you want to add something to it.

        Examples
        --------
        >>> my_profile = Profile(5, 3)
        >>> _ = my_profile.uniform_distribution(100)
        >>> my_profile.plot_profile("3D", show=False)
        <matplotlib.axes._subplots.Axes3DSubplot object at ...>
        >>> my_profile.plot_profile("ternary", show=False)
        TernaryAxesSubplot: ...
        """

        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        if plot_kind == "3D":
            ax = self._plot_profile_3D(fig, dim, position)
        elif plot_kind == "ternary":
            ax = self._plot_profile_ternary(fig, dim, position)
        else:
            raise ValueError("plot_kind should '3D' or 'ternary'")
        ax.set_title("Profile of voters (%i,%i,%i)" % (dim[0], dim[1], dim[2]), fontsize=24)
        if show:
            plt.show()
        return ax

    def _plot_scores_3D(self, sizes, fig, position, dim):
        """
        Plot a figure of the profile on a 3D space with the embeddings vector having the sizes passed
        as parameters.

        Parameters
        ----------
        sizes : np.ndarray
            The norm of the vectors. Should be of length :attr:`n_voters`.
        fig : matplotlib figure
            The figure on which we add the plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.
        dim : list
            The 3 dimensions we are using for our plot.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure, if you want to add something to it.

        """
        ax = create_3D_plot(fig, position)
        for i, (v, s) in enumerate(zip(self.embeddings, sizes)):
            x1 = v[dim[0]]
            x2 = v[dim[1]]
            x3 = v[dim[2]]
            ax.plot([0, s * x1], [0, s * x2], [0, s * x3], color=(x1 * 0.8, x2 * 0.8, x3 * 0.8), alpha=0.4)

        return ax

    def _plot_scores_ternary(self, sizes, fig, position, dim):
        """
        Plot a figure of the profile on a 2D space with the voters dots having the sizes passed
        as parameters.

        Parameters
        ----------
        sizes : np.ndarray
            The size of the dots. Should be of length :attr:`n_voters`.
        fig : matplotlib figure
            The figure on which we add the plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.
        dim : list
            The 3 dimensions we are using for our plot.

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure, if you want to add something to it.

        """
        tax = create_ternary_plot(fig, position)
        for i, (v, s) in enumerate(zip(self.embeddings, sizes)):
            x1 = v[dim[0]]
            x2 = v[dim[2]]
            x3 = v[dim[1]]
            vec = [x1, x2, x3]
            tax.scatter([normalize(vec)**2], color=(x1 * 0.8, x3 * 0.8, x2 * 0.8), alpha=0.7, s=max(s * 50, 1))

        return tax

    def plot_scores(self, sizes, title="", plot_kind="3D", dim=None, fig=None, position=None, show=True):
        """
        Plot a figure of the profile on a 2D space with the voters dots having the sizes passed
        as parameters.

        Parameters
        ----------
        sizes : np.ndarray
            The score given by each voter. Should be of length :attr:`n_voters`.
        title : str
            Title of the figure
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        fig : matplotlib figure
            The figure on which we add the plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.
        dim : list
            The 3 dimensions we are using for our plot.
        show : bool
            If True, show the figure at the end of the function

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure, if you want to add something to it.

        """
        if dim is None:
            dim = [0, 1, 2]
        else:
            if len(dim) != 3:
                raise ValueError("The number of dimensions should be 3")

        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        if plot_kind == "3D":
            ax = self._plot_scores_3D(sizes, fig, position, dim)
        elif plot_kind == "ternary":
            ax = self._plot_scores_ternary(sizes, fig, position, dim)
        else:
            raise ValueError("plot_kind should '3D' or 'ternary'")

        ax.set_title(title, fontsize=16)
        if show:
            plt.show()

        return ax

    def plot_candidate(self, candidate, plot_kind="3D", dim=None, fig=None, position=None, show=True):
        """
        Plot a figure of the profile on a 2D space with the voters dots having the scores they give to the
        candidate passed as a parameter.

        Parameters
        ----------
        candidate : int
            The candidate for which we want to show the profile. Should be < :attr:`n_candidates`.
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        fig : matplotlib figure
            The figure on which we add the plot.
        position : list
            The position of the plot on the figure. Should be of the form
            `[n_rows, n_columns, position]`.
        dim : list
            The 3 dimensions we are using for our plot.
        show : bool
            If True, show the figure at the end of the function

        Return
        ------
        matplotlib ax
            The matplotlib ax with the figure, if you want to add something to it.

        """
        return self.plot_scores(self.scores[::, candidate],
                                title="Candidate #%i" % (candidate + 1),
                                plot_kind=plot_kind,
                                dim=dim,
                                fig=fig,
                                position=position,
                                show=show)

    def plot_candidates(self, plot_kind="3D", dim=None, list_candidates=None, list_titles=None, row_size=5, show=True):
        """
        Plot the profile of the voters for every candidates or a list of candidate,
        using the scores given by the voters. The plot is either on a 3D plot, or on a ternary plot.
        Only three dimensions are represented.

        Parameters
        ----------
        plot_kind : str
            The kind of plot we want to show. Can be "3D" or "ternary".
        dim : list
            The 3 dimensions we are using for our plot.
        list_candidates : int list
            The list of candidates we want to plot. Should contains integer lower than
            :attr:`n_candidates`. Default is range(:attr:`n_candidates`).
        list_titles : str list
            Contains the title of the plots.Should be the same length than list_candidates.
        row_size : int
            Number of subplots by row. Default is set to 5.
        show : bool
            If True, plot the figure at the end of the function.

        Examples
        --------
        >>> my_profile = Profile(5, 3)
        >>> _ = my_profile.uniform_distribution(100)
        >>> my_profile.plot_candidates("3D", show=False)
        >>> my_profile.plot_candidates("ternary", show=False)
        """

        if list_candidates is None:
            list_candidates = range(self.n_candidates)
        if list_titles is None:
            list_titles = ["Candidate #%i" % c for c in list_candidates]
        else:
            list_titles = ["%s " % t for t in list_titles]

        n_candidates = len(list_candidates)
        n_rows = (n_candidates - 1) // row_size + 1
        fig = plt.figure(figsize=(5 * row_size, n_rows * 5))
        position = [n_rows, row_size, 1]
        for candidate, title in (zip(list_candidates, list_titles)):
            self.plot_scores(self.scores[::, candidate],
                             title=title,
                             plot_kind=plot_kind,
                             dim=dim,
                             fig=fig,
                             position=position,
                             show=False)
            position[2] += 1

        if show:
            plt.show()

    def copy(self):
        """
        Return a copy of the profile

        Return
        ------
        Profile
            A copy of this profile

        Examples
        --------
        >>> my_profile = Profile(2, 3)
        >>> _ = my_profile.add_voter([.1, .2, .5], [.5, .8])
        >>> second_profile = my_profile.copy()
        >>> _ = second_profile.add_voter([.1, .2, .5], [.5, .8])
        >>> my_profile.n_voters
        1
        >>> second_profile.n_voters
        2

        """
        p = Profile(self.n_candidates, self.n_dim)
        p.embeddings = self.embeddings
        p.n_voters = self.n_voters
        p.scores = self.scores
        return p
