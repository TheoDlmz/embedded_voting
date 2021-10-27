# -*- coding: utf-8 -*-
"""
Copyright Théo Delemazure
theo.delemazure@ens.fr

This file is part of Embedded Voting.
"""
import numpy as np
import matplotlib.pyplot as plt
from embedded_voting.utils.cached import DeleteCacheMixin
from embedded_voting.embeddings.embeddings import Embeddings


class Profile(DeleteCacheMixin):
    """
    A profile of voter with embeddings.

    Parameters
    ----------
    ratings : np.ndarray
        The ratings given by the voters to the candidates.
        Its dimensions are :attr:`n_voters`, :attr:`n_candidates`.
    embeddings : Embeddings
        The embeddings of the voters. If none is specified, embeddings are the identity matrix.

    Attributes
    ----------
    n_voters : int
        The number of voters in the profile.
    n_candidates : int
        The number of candidates in this profile.
    embeddings : Embeddings
        The embeddings of the voters.
    ratings : np.ndarray
        The ratings given by the voters to the candidates.
        Its dimensions are :attr:`n_voters`, :attr:`n_candidates`.

    Examples
    --------
    >>> profile = Profile(np.ones((100,10)))
    >>> profile.n_candidates
    10
    >>> profile.n_voters
    100
    >>> profile.ratings[-1]
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    """

    def __init__(self, ratings, embeddings=None):
        self.ratings = ratings
        self.n_voters, self.n_candidates = ratings.shape
        if embeddings is not None:
            if embeddings.n_voters != ratings.shape[0]:
                raise ValueError("ratings and embeddings should have the same first dimension")
        else:
            embeddings = Embeddings(np.eye(self.n_voters))
        self.embeddings = embeddings

    def set_embeddings(self, embeddings):
        """
        A function to properly set or update embeddings of the voters

        Parameters
        ----------
        embeddings: Embeddings
            The new embeddings of the voters

        Return
        ------
        Profile
            The profile itself

        Examples
        --------
        >>> profile = Profile(np.random.rand(3,10))
        >>> embs = Embeddings(np.array([[.5,.9,.4],[.4,.7,.5],[.4,.2,.4]]), norm=False)
        >>> profile.set_embeddings(embs)
        <embedded_voting.profile.profile.Profile object at ...>
        >>> profile.embeddings
        <embedded_voting.embeddings.embeddings.Embeddings object at ...>
        >>> profile.embeddings.positions
        array([[0.5, 0.9, 0.4],
               [0.4, 0.7, 0.5],
               [0.4, 0.2, 0.4]])
        """
        if embeddings.n_voters != self.ratings.shape[0]:
            raise ValueError("ratings and embeddings should have the same first dimension")
        self.embeddings = embeddings
        return self

    def generate_embeddings(self, embedder):
        """
        A function to properly generate or update embeddings of the voters

        Parameters
        ----------
        embedder: Embedder
            The embedder

        Return
        ------
        Profile
            The profile itself

        Examples
        --------
        """
        self.embeddings = embedder(self.ratings)
        return self

    def standardize(self, cut_one=True):
        """
        Standardize the ratings
        between the different voters.

        Parameters
        ----------
        cut_one : bool
            if True, then the maximum score is `1`.
            Every scores above `1` will be set to `1`.
            The minimum score is always 0.

        Return
        ------
        Profile
            The profile itself.

        Examples
        --------
        >>> profile = Profile(np.array([[.6, .8, 1],[.4, .5, .3]]))
        >>> profile.ratings.mean(axis=0)
        array([0.5 , 0.65, 0.65])
        >>> profile.standardize().ratings.mean(axis=0)
        array([0.25, 0.75, 0.5 ])
        """

        mu = self.ratings.mean(axis=1)
        sigma = self.ratings.std(axis=1)
        new_ratings = self.ratings.T - np.tile(mu, (self.n_candidates, 1))
        new_ratings = new_ratings / sigma
        new_ratings += 1
        new_ratings /= 2
        new_ratings = np.maximum(new_ratings, 0)
        if cut_one:
            new_ratings = np.minimum(new_ratings, 1)
        self.ratings = new_ratings.T

        return self

    def plot_candidate(self, candidate, plot_kind="3D", dim=None, fig=None, plot_position=None, show=True):
        """
        Plot a figure of the profile
        with the voters having the ratings they give
        to the `candidate` passed as parameters
        as size.

        Parameters
        ----------
        candidate : int
            The candidate for which we
            want to show the profile.
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
        return self.embeddings.plot_scores(self.ratings[::, candidate],
                                           title="Candidate %i" % (candidate + 1),
                                           plot_kind=plot_kind,
                                           dim=dim,
                                           fig=fig,
                                           plot_position=plot_position,
                                           show=show)

    def plot_candidates(self, plot_kind="3D", dim=None, list_candidates=None, list_titles=None, row_size=5, show=True):
        """
        Plot the profile of the voters
        for every candidate or a list of candidates,
        using the ratings given by the voters as size for
        the voters. The plot is either on a 3D plot,
        or on a ternary plot.
        Only three dimensions can be represented.

        Parameters
        ----------
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
            list_candidates = range(self.n_candidates)
        if list_titles is None:
            list_titles = ["Candidate %i" % c for c in list_candidates]
        else:
            list_titles = ["%s " % t for t in list_titles]

        n_candidates = len(list_candidates)
        n_rows = (n_candidates - 1) // row_size + 1
        fig = plt.figure(figsize=(5 * row_size, n_rows * 5))
        position = [n_rows, row_size, 1]
        for candidate, title in (zip(list_candidates, list_titles)):
            self.embeddings.plot_scores(self.ratings[::, candidate],
                                        title=title,
                                        plot_kind=plot_kind,
                                        dim=dim,
                                        fig=fig,
                                        plot_position=position,
                                        show=False)
            position[2] += 1

        if show:
            plt.show()  # pragma: no cover

    def scored_embeddings(self, candidate, square_root=True):
        """
        Return the embeddings matrix, but
        each voter's embedding is multiplied
        by the score it gives to the candidate
        passed as parameter.

        Parameters
        ----------
        candidate : int
            The candidate of whom we use the scores.
        square_root : bool
            If True, we multiply by the square root
            of the scores given to the candidate
            instead of the scores themself.

        Return
        ------
        np.ndarray
            The embedding matrix, of shape :attr:`n_voters`, :attr:`n_dim`.

        Examples
        --------
        >>> embs = Embeddings(np.array([[0, .5, .5],[.5, .5, 0]])).normalize()
        >>> profile = Profile(np.array([[.5], [.2]]), embs)
        >>> profile.embeddings.positions
        array([[0.        , 0.70710678, 0.70710678],
               [0.70710678, 0.70710678, 0.        ]])
        >>> profile.scored_embeddings(0)
        array([[0.        , 0.5       , 0.5       ],
               [0.31622777, 0.31622777, 0.        ]])
        >>> profile.scored_embeddings(0, square_root=False)
        array([[0.        , 0.35355339, 0.35355339],
               [0.14142136, 0.14142136, 0.        ]])
        """

        embeddings = []
        for i in range(self.n_voters):
            if square_root:
                s = np.sqrt(self.ratings[i, candidate])
            else:
                s = self.ratings[i, candidate]
            embeddings.append(self.embeddings.positions[i] * s)
        return np.array(embeddings)

'''
    def fake_covariance_matrix(self, candidate, f, square_root=True):
        """
        This function return a matrix `M`
        such that for all voters `i`  and `j`,
        `M[i,j] = scores[i, candidate] * scores[j, candidate]
        * f(embeddings[i], embeddings[j])`
        (cf :attr:`scores`, :attr:`embeddings`).

        Parameters
        ----------
        candidate : int
            The candidate for whom we want the matrix.
        f : callable
            Similarity function between
            two embeddings vector of length :attr:`n_dim`.
            Input : (np.ndarray, np.ndarray).
            Output : float.
        square_root : bool
            If True, we multiply by the square root
            of the scores given to the candidate
            instead of the scores themself.

        Return
        ------
        np.ndarray
            Matrix of shape :attr:`n_voters`, :attr:`n_voters`.

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

    '''
