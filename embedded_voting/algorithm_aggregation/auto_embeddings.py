from embedded_voting.profile.Profile import Profile
import numpy as np


class AutoProfile(Profile):
    """
    A subclass of :class:`Profile` that can automatically generate embeddings
    using the scores and some training samples.
    """

    def add_voters_auto(self, scores,  samples=None, n_dim=0, normalize_score=True):
        """
        This function will create embeddings for the voters based on the scores
        they gave to the candidate and to other input passed as training samples
        with the parameters `samples`. This function use the eigenvalues of the
        covariance matrix of the samples to create voters' embeddings.

        Parameters
        ----------
        scores : np.ndarray or list
            The scores that the voters give to the candidates.
            Should be of shape :attr:`~embedded_voting.Profile.n_voters`,
            :attr:`~embedded_voting.Profile.n_candidates`.
        samples : np.ndarray or list
            Additional training samples that the function will use for
            the covariance matrix. Should be of shape _, :attr:`~embedded_voting.Profile.n_candidates`.
        n_dim : int
            The number of dimension we want in our profile.
            If it is 0, the number of dimension is inferred with the eigenvalues.
            By default, it is set to 0.
        normalize_score : bool
            If True, normalize the scores to `[0, 1]` range at the end of the function.

        Return
        ------
        AutoProfile
            The object itself

        Examples
        --------
        >>> np.random.seed(42)
        >>> profile = AutoProfile(50, 3)
        >>> score = [list(np.random.rand(50))]*5 + [list(np.random.rand(50))]*5
        >>> profile.add_voters_auto(score)
        <embedded_voting.algorithm_aggregation.auto_embeddings.AutoProfile object at ...>
        >>> profile.n_dim
        2
        >>> profile.embeddings
        array([[-0.52310182, -0.85227019],
               [-0.52310182, -0.85227019],
               [-0.52310182, -0.85227019],
               [-0.52310182, -0.85227019],
               [-0.52310182, -0.85227019],
               [-0.29009365,  0.95699826],
               [-0.29009365,  0.95699826],
               [-0.29009365,  0.95699826],
               [-0.29009365,  0.95699826],
               [-0.29009365,  0.95699826]])
        """
        scores = np.array(scores)
        if samples is None:
            samples_total = scores
        else:
            samples = np.array(samples)
            samples_total = np.concatenate([samples, scores], axis=1)
        cov = np.cov(samples_total)

        if n_dim == 0:
            eigen_values, _ = np.linalg.eig(cov)
            eigen_values = np.sort(eigen_values)[::-1]
            eigen_values_participation = eigen_values / np.sum(eigen_values)
            while n_dim < len(eigen_values_participation) and eigen_values_participation[n_dim] > 0.01:
                n_dim += 1

        cov_transpose = np.cov(samples_total.T)
        _, eigen_vectors = np.linalg.eig(cov_transpose)
        projection_matrix = (eigen_vectors.T[:][:n_dim]).T
        embs = samples_total.dot(projection_matrix)

        self.n_dim = n_dim
        self.embeddings = np.zeros((0, n_dim))

        if normalize_score:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        embs = np.real(embs)
        self.add_voters(embs, scores)
        return self

    def add_voters_cov(self, scores, samples=None, normalize_score=False):
        """
        This function will create embeddings for the voters based on the scores
        they gave to the candidate and to other input passed as training samples
        with the parameters `samples`. This function directly use of the
        covariance matrix of the samples to create voters' embeddings.

        Parameters
        ----------
        scores : np.ndarray or list
            The scores that the voters give to the candidates.
            Should be of shape :attr:`~embedded_voting.Profile.n_voters`,
            :attr:`~embedded_voting.Profile.n_candidates`.
        samples : np.ndarray or list
            Additional training samples that the function will use for
            the covariance matrix. Should be of shape _, :attr:`~embedded_voting.Profile.n_candidates`.
        normalize_score : bool
            If True, normalize the scores to `[0, 1]` range at the end of the function.

        Return
        ------
        AutoProfile
            The object itself

        Examples
        --------
        >>> np.random.seed(42)
        >>> profile = AutoProfile(50, 4)
        >>> score = [list(np.random.rand(50))]*2 + [list(np.random.rand(50))]*2
        >>> profile.add_voters_cov(score)
        <embedded_voting.algorithm_aggregation.auto_embeddings.AutoProfile object at ...>
        >>> profile.n_dim
        4
        >>> profile.embeddings
        array([[0.0834535 , 0.0834535 , 0.00551433, 0.00551433],
               [0.0834535 , 0.0834535 , 0.00551433, 0.00551433],
               [0.00551433, 0.00551433, 0.09415171, 0.09415171],
               [0.00551433, 0.00551433, 0.09415171, 0.09415171]])
        """
        scores = np.array(scores)
        if samples is None:
            samples_total = scores
        else:
            samples = np.array(samples)
            samples_total = np.concatenate([samples, scores], axis=1)
        cov = np.cov(samples_total)

        if normalize_score:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        self.add_voters(cov, scores, normalize_embs=False)
        return self
