import numpy as np
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_covariance import EmbeddingsFromRatingsCovariance
from embedded_voting.ratings.ratings_generator_epistemic_multivariate import RatingsGeneratorEpistemicMultivariate
from embedded_voting.rules.singlewinner_rules.rule import Rule
from embedded_voting.utils.cached import cached_property
from embedded_voting.utils.miscellaneous import clean_zeros, pseudo_inverse_scalar


class RuleMLEGaussian(Rule):
    """
    A rule that computes the scores of the candidates, assuming that the embeddings of the voters correspond to a
    covariance matrix.

    For this rule, the embeddings must be a matrix `n_voters` * `n_voters`.

    Examples
    --------
    Consider a generating epistemic model, where the true value of each candidate is uniformly drawn in a given
    interval, and where the voters add a noise which is multivariate Gaussian.

    >>> np.random.seed(42)
    >>> covariance_matrix = np.array([
    ...     [2.02, 1.96, 0.86, 0.81, 1.67],
    ...     [1.96, 3.01, 1.46, 0.69, 1.59],
    ...     [0.86, 1.46, 0.94, 0.39, 0.7 ],
    ...     [0.81, 0.69, 0.39, 0.51, 0.9 ],
    ...     [1.67, 1.59, 0.7 , 0.9 , 1.78]
    ... ])
    >>> ratings_generator = RatingsGeneratorEpistemicMultivariate(covariance_matrix=covariance_matrix)
    >>> ratings = ratings_generator(n_candidates=2)
    >>> ratings_generator.ground_truth_
    array([17.73956049, 14.3887844 ])
    >>> ratings
    Ratings([[17.56232759, 14.51592899],
             [16.82544972, 15.78818081],
             [17.51952581, 14.44449175],
             [17.34964888, 14.4010885 ],
             [16.69480298, 14.9281998 ]])

    If we know the covariance matrix of the noises, then `RuleMLEGaussian` is the maximum likelihood
    estimator of the ground truth:

    >>> election = RuleMLEGaussian()(ratings, embeddings=covariance_matrix)
    >>> election.scores_ # doctest: +ELLIPSIS
    [268.6683142..., 221.5083075...]


    """

    def __init__(self, embeddings_from_ratings=None, tol=1e-6):
        self.tol = tol
        if embeddings_from_ratings is None:
            embeddings_from_ratings = EmbeddingsFromRatingsCovariance()
        super().__init__(score_components=1, embeddings_from_ratings=embeddings_from_ratings)

    @cached_property
    def pinv_covariance_(self):
        tol = self.tol
        n, m = self.embeddings_.shape
        min_d = min(n, m)
        u, s, v = np.linalg.svd(self.embeddings_)
        clean_zeros(s, tol=tol)
        dia = np.zeros((m, n))
        dia[:min_d, :min_d] = np.diag([pseudo_inverse_scalar(e) for e in s])
        inverse = v.T @ dia @ u.T
        clean_zeros(inverse, tol=tol)
        return inverse

    @cached_property
    def weights_(self):
        return self.pinv_covariance_.sum(axis=0)

    def _score_(self, candidate):
        return float(self.ratings_.candidate_ratings(candidate) @ self.weights_)


"""
    #>>> np.linalg.norm(ratings_generator.ground_truth_ - election.scores_)  # Error estimation
    #0.4783006898563199

    As a baseline, consider the error for the naive arithmetic mean:

    #>>> scores_average = np.mean(ratings, axis=0)
    #>>> np.linalg.norm(ratings_generator.ground_truth_ - scores_average, 2)
    0.6911799682033576

    However, in practice, we often do not know the covariance matrix of the noise. A workaround can be to
    use the covariance matrix of the ratings:

    #>>> embeddings = EmbeddingsFromRatingsCovariance()(ratings)
    #>>> election = RuleMLEGaussian()(ratings, embeddings)
    #>>> election.scores_  # doctest: +ELLIPSIS
    #[12.90546983325..., 19.502265626617...]

    Actually, this is the default behavior of `RuleMLEGaussian` when no embeddings are given:

    #>>> election = RuleMLEGaussian()(ratings)
    #>>> election.scores_  # doctest: +ELLIPSIS
    #[12.90546983325..., 19.502265626617...]

    Unfortunately, this approximation is relevant if there are a large number of candidates (to have a good estimation
    of the covariance matrix) and if the noise is large compared to the differences between true values (so
    that the covariance of ratings approximates well the covariance of noises), which is not a common case. In our
    example, the assumptions are not met, and the result is not even as good as the naive arithmetic mean:

    #>>> np.linalg.norm(ratings_generator.ground_truth_ - election.scores_)  # Error estimation
    #0.839945516610...
    
"""