# -*- coding: utf-8 -*-
"""
This file is part of Embedded Voting.
"""

import numpy as np


def normalize(x):
    """
    Normalize the input vector.

    Parameters
    ----------
    x : np.ndarray or list

    Return
    ------
    np.ndarray
        `x` divided by its Euclidean norm.

    Examples
    --------
    >>> my_vector = np.arange(3)
    >>> normalize(my_vector)
    array([0.        , 0.4472136 , 0.89442719])

    >>> my_vector = [0, 1, 2]
    >>> normalize(my_vector)
    array([0.        , 0.4472136 , 0.89442719])

    If `x` is null, then `x` is returned (only case where the result has not a norm of 1):

    >>> my_vector = [0, 0, 0]
    >>> normalize(my_vector)
    array([0, 0, 0])
    """
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.array(x)
    return x / norm


def center_and_normalize(x):
    """
    Center and normalize the input vector.

    Parameters
    ----------
    x : np.ndarray or list

    Returns
    -------
    np.ndarray
        `x` minus its mean. Then the result is normalized (divided by its norm).

    Examples
    --------
    >>> my_vector = [0, 1, 2]
    >>> center_and_normalize(my_vector)
    array([-0.70710678,  0.        ,  0.70710678])
    """
    return normalize(x - np.mean(x))


def max_angular_dilatation_factor(vector, center):
    """
    Maximum angular dilatation factor to stay in the positive orthant.

    Consider `center` and `vector` two unit vectors of the positive orthant. Consider a "spherical dilatation" that
    moves `vector` by multiplying the angle between `center` and `vector` by a given dilatation factor. The question is:
    what is the maximal value of this dilatation factor so that the result still is in the positive orthant?

    More formally, there exists a unit vector `unit_orthogonal` and an angle `theta in [0, pi/2]` such that
    `vector = cos(theta) * center + sin(theta) * unit_orthogonal`. Then there exists a maximal angle
    `theta_max in [0, pi/2]` such that `cos(theta_max) * center + sin(theta_max) * unit_orthogonal`
    is still in the positive orthant. We define the maximal angular dilatation factor as `theta_max / theta`.

    Parameters
    ----------
    vector : np.ndarray
        A unit vector in the positive orthant.
    center : np.ndarray
        A unit vector in the positive orthant.

    Returns
    -------
    float
        The maximal angular dilatation factor. If `vector` is equal to `center`, then `np.inf` is returned.

    Examples
    --------
    >>> max_angular_dilatation_factor(
    ...     vector=np.array([1, 0]),
    ...     center=np.array([1, 1]) * np.sqrt(1/2)
    ... )
    1.0
    >>> max_angular_dilatation_factor(
    ...     vector=np.array([1, 1]) * np.sqrt(1/2),
    ...     center=np.array([1, 0])
    ... )
    2.0

    >>> my_center = np.array([1., 1., 1.]) * np.sqrt(1/3)
    >>> my_unit_orthogonal = np.array([1, -1, 0]) * np.sqrt(1/2)
    >>> my_theta = np.pi / 9
    >>> my_vector = np.cos(my_theta) * my_center + np.sin(my_theta) * my_unit_orthogonal
    >>> k = max_angular_dilatation_factor(vector=my_vector, center=my_center)
    >>> k
    1.961576024...
    >>> dilated_vector = np.cos(k * my_theta) * my_center + np.sin(k * my_theta) * my_unit_orthogonal
    >>> np.round(dilated_vector, 4)
    array([0.8944, 0.    , 0.4472])

    >>> max_angular_dilatation_factor(
    ...     np.array([np.sqrt(1/2), np.sqrt(1/2)]),
    ...     np.array([np.sqrt(1/2), np.sqrt(1/2)])
    ... )
    inf
    """
    scalar_product = vector @ center
    if 1 < scalar_product <= 1.0001:  # Compensate for numerical approximations.
        scalar_product = 1.
    theta = np.arccos(scalar_product)
    vector_collinear = scalar_product * center  # Component of `vector` on the direction of `center`.
    vector_orthogonal = vector - vector_collinear  # Component of `vector` in the orthogonal direction.
    norm_vector_orthogonal = np.linalg.norm(vector_orthogonal)
    if theta == 0 or norm_vector_orthogonal == 0:
        return np.inf
    unit_orthogonal = vector_orthogonal / norm_vector_orthogonal
    mask = unit_orthogonal < 0
    if not any(mask):
        theta_max = np.pi / 2
    else:
        theta_max = np.arctan(np.min(- center[mask] / unit_orthogonal[mask]))
    return theta_max / theta


def ranking_from_scores(scores):
    """
    Deduce ranking over the candidates from their scores.

    Parameters
    ----------
    scores: list
        List of floats, or list of tuple.

    Returns
    -------
    ranking: list
        The indices of the candidates, so candidate `ranking[0]` has the best score, etc.
        If scores are floats, higher scores are better. If scores are tuples, a lexicographic
        order is used. In case of tie, candidates with lower indices are favored.

    Examples
    --------
    >>> my_scores = [4, 1, 3, 4, 0, 2, 1, 0, 1, 0]
    >>> ranking_from_scores(my_scores)
    [0, 3, 2, 5, 1, 6, 8, 4, 7, 9]

    >>> my_scores = [(1, 0, 3), (2, 1, 5), (0, 1, 1), (2, 1, 4)]
    >>> ranking_from_scores(my_scores)
    [1, 3, 0, 2]
    """
    return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)


def winner_from_scores(scores):
    """
    Deduce the best of candidates from their scores.

    Parameters
    ----------
    scores: list
        List of floats, or list of tuple.

    Returns
    -------
    winner: int
        The index of the winning candidate. If scores are floats, higher scores are better. If scores are tuples,
        a lexicographic order is used. In case of tie, candidates with lower indices are favored.

    Examples
    --------
    >>> my_scores = [4, 1, 3, 4, 0, 2, 1, 0, 1, 0]
    >>> winner_from_scores(my_scores)
    0

    >>> my_scores = [(1, 0, 3), (2, 1, 5), (0, 1, 1), (2, 1, 4)]
    >>> winner_from_scores(my_scores)
    1
    """
    return max(range(len(scores)), key=scores.__getitem__)


def volume_parallelepiped(matrix):
    """Volume of the parallelepiped defined by the rows of a matrix.

    Parameters
    ==========
    matrix: np.ndarray
        The matrix.

    Returns
    =======
    float
        The volume of the parallelepiped defined by the rows of a matrix (in the `r`-dimensional space defined
        by its `r` rows). If the rank of the matrix is less than its number of rows, then the result is 0.

    Examples
    ========
    >>> volume_parallelepiped(matrix=np.array([[10, 0, 0, 0], [0, 42, 0, 0]]))  # doctest: +ELLIPSIS
    420.0...

    >>> volume_parallelepiped(matrix=np.array([[10, 0, 0, 0], [42, 0, 0, 0]]))
    0.0

    >>> volume_parallelepiped(matrix=np.array([[10, 0, 0, 0]]))  # doctest: +ELLIPSIS
    10.0...
    """
    det = np.linalg.det(matrix @ matrix.T)
    det = max(det, 0)  # In theory, the determinant should be nonnegative, but there are numerical issues
    return np.sqrt(det)


def singular_values_short(matrix):
    """
    Singular values of a matrix (short version).

    Parameters
    ----------
    matrix: np.ndarray

    Returns
    -------
    np.ndarray
        Singular values of the matrix. In order to have a "short" version (and limit computation), we consider
        the square matrix of smallest dimensions among `matrix @ matrix.T` and `matrix.T @ matrix`, and then output
        the square roots of its eigenvalues.

    Examples
    --------
    >>> my_matrix = np.array([
    ...     [0.2 , 0.5 , 0.7 , 0.9 , 0.4 ],
    ...     [0.1 , 0.  , 1.  , 0.8 , 0.8 ],
    ...     [0.17, 0.4 , 0.66, 0.8 , 0.4 ]
    ... ])
    >>> singular_values = singular_values_short(my_matrix)
    >>> np.round(singular_values, 4)
    array([2.2747, 0.5387, 0.    ])
    """
    r, c = matrix.shape
    if r < c:
        square_matrix = matrix @ matrix.T
    else:
        square_matrix = matrix.T @ matrix
    eigenvalues = np.linalg.eigvals(square_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    # In theory, all eigenvalues are nonnegative, but due to numerical approximations, it may happen that
    # some of them are computed as very small negative values. Hence the np.maximum().

    return np.sort(np.sqrt(eigenvalues))[::-1]


def pseudo_inverse_scalar(x):
    """
    Parameters
    ----------
    x: :class:`float`

    Returns
    -------
    :class:`float`
        Inverse of x if it is not 0.

    Examples
    --------

    >>> pseudo_inverse_scalar(2.0)
    0.5
    >>> pseudo_inverse_scalar(0)
    0.0
    """
    return 0.0 if x == 0 else 1 / x



def clean_zeros(matrix, tol=1e-10):
    """
    Replace in-place all small values of a matrix by 0.

    Parameters
    ----------
    matrix: :class:`~numpy.ndarray`
        Matrix to clean.
    tol: :class:`float`, optional
        Threshold. All entries with absolute value lower than `tol` are put to zero.

    Returns
    -------
    None

    Examples
    --------

    >>> import numpy as np
    >>> mat = np.array([[1e-12, -.3], [.8, -1e-13]])
    >>> clean_zeros(mat)
    >>> mat # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. , -0.3],
           [ 0.8,  0. ]])
    """
    matrix[abs(matrix[:]) < tol] = 0
