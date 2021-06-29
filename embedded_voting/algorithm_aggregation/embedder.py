import numpy as np
from embedded_voting.profile.Profile import Profile


class Embedder(Profile):
    """
    A class for a Profile that used for several elections, with different candidates
    in each election. The Embedder keeps in memory the scores given by the voters during
    each election.

    Parameters
    ----------
    scores: np.ndarray or list
        The scores of the candidates in the first elections.
        ``scores[i,j]`` corresponds to the
        score given by the voter i to candidate j.

    Attributes
    ----------
    n_voters : int
        The number of voters in the profile.
    n_candidates : int
        The number of candidates in this profile.
    n_dim : int
        The number of dimensions of the voters' embeddings.
    embeddings : np.ndarray
        The scores given by the candidate during every elections.
    scores : np.ndarray
        The scores given by the voters to the candidates.
        Its dimensions are :attr:`n_voters`, :attr:`n_candidates`.
    fast_embeddings : np.ndarray
        The preprocessed matrix used in aggregation rules of the :class:`~embedded_voting.Fast` family.
        This is updated everytime we use the training method.
    n_sing_val : int
        The number of singular values that should be used in aggregation rules
        of the :class:`~embedded_voting.Fast` family.
        This is updated everytime we use the training method.

    Examples
    --------
    >>> embedder = Embedder([[7, 5, 9, 6, 1], [7, 5, 9, 5, 2], [6, 4, 2, 4, 4], [3, 8, 1, 7, 8]])
    >>> embedder.n_voters
    4
    >>> embedder.n_candidates
    5
    >>> embedder([[2, 4, 8], [9, 2, 1], [0, 2, 5], [4, 5, 3]]).n_candidates
    3
    >>> embedder.n_sing_val
    2
    >>> embedder.train().n_sing_val
    1

    """

    def __init__(self, scores):
        scores = np.array(scores)
        n_voters, n_candidates = scores.shape
        super().__init__(n_candidates, n_voters)
        self.embeddings = scores
        self.scores = scores
        self.n_candidates = n_candidates
        self.n_dim = n_candidates
        self.n_voters = n_voters
        self.fast_embeddings = None
        self.n_sing_val = 0
        self.train()

    def __call__(self, new_scores):
        """
        Run an election on the scores given.

        Parameters
        ----------
        new_scores: np.ndarray or list
            The matrix of scores of the new election.

        Return
        ------
        Embedder
            The object.

        """
        new_scores = np.array(new_scores)
        n_voters, n_candidates = new_scores.shape
        self.n_candidates = n_candidates
        self.n_dim += n_candidates
        self.embeddings = np.concatenate([self.embeddings, new_scores], axis=1)
        self.scores = new_scores
        return self

    def train(self):
        """
        This function train the embedder on the newest data to identify dependencies
        between the voters. It updates :attr:`self.fast_embeddings` and :attr:`n_sing_val`

        Return
        ------
        Embedder
            The object.

        """
        embeddings = self.embeddings
        embeddings = (embeddings.T / np.sqrt((embeddings ** 2).sum(axis=1))).T

        u, s, v = np.linalg.svd(embeddings)
        n_voters, n_candidates = embeddings.shape
        s = np.sqrt(s)
        s /= s.sum()
        n_v = 0
        for s_e in s:
            if s_e >= max(1 / n_voters, 1 / n_candidates):
                n_v += 1
        self.n_sing_val = n_v

        self.fast_embeddings = np.dot(embeddings, embeddings.T)
        return self




