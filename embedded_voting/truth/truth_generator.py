class TruthGenerator:
    """
    A generator for the ground truth ("true value") of each candidate.
    """

    def __call__(self, n_candidates):
        """
        Generate the true values of the candidates.

        Parameters
        ----------
        n_candidates : int
            The number of candidates of which we want the true values.

        Returns
        -------
        true_values : np.ndarray
            The true value of each candidate. Size: :attr:`n_candidates`.
        """
        raise NotImplementedError
