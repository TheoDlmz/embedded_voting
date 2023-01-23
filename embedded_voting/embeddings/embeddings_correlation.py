from embedded_voting.embeddings.embeddings import Embeddings


# noinspection PyUnresolvedReferences
class EmbeddingsCorrelation(Embeddings):
    """Embeddings based on correlation, dedicated to :class:`RuleFast`.

    Parameters
    ----------
    positions : np.ndarray or list or Embeddings
        The embeddings of the voters. Its dimensions are :attr:`n_voters`, :attr:`n_dim`.
    n_sing_val : int
        "Effective" number of singular values.
    ratings_means : np.ndarray
        Mean rating for each voter.
    ratings_stds : np.ndarray
        Standard deviation of the ratings for each voter.
    norm: bool
        If True, normalize the embeddings.

    Examples
    --------
    >>> embeddings = EmbeddingsCorrelation([[1, 2], [3, 4]], n_sing_val=2, ratings_means=[.1, .2],
    ...                                    ratings_stds=[.3, .4], norm=True)
    >>> embeddings
    EmbeddingsCorrelation([[0.4472136 , 0.89442719],
                           [0.6       , 0.8       ]])
    >>> embeddings.n_sing_val
    2
    >>> embeddings.ratings_means
    [0.1, 0.2]

    >>> embeddings2 = embeddings.copy()
    >>> embeddings2.n_sing_val
    2
    """

    def __new__(cls, positions, n_sing_val, ratings_means, ratings_stds, norm):
        obj = super().__new__(cls, positions=positions, norm=norm)
        obj.n_sing_val = n_sing_val
        obj.ratings_means = ratings_means
        obj.ratings_stds = ratings_stds
        return obj
