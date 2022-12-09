"""
This file is part of Embedded Voting.
"""
import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings_from_ratings.embeddings_from_ratings_identity import EmbeddingsFromRatingsIdentity
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property


class MultiwinnerRule(DeleteCacheMixin):
    """
    A class for multiwinner rules, in other words
    aggregation rules that elect a committee of
    candidates of size :attr:`k_`, given a
    ratings of voters with embeddings.

    Parameters
    ----------
    k : int
        The size of the committee.

    Attributes
    ----------
    ratings: np.ndarray
        The ratings given by voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    k_ : int
        The size of the committee.
    """

    def __init__(self,  k=None):
        self.ratings = None
        self.embeddings = None
        self.k_ = k

    def __call__(self, ratings, embeddings=None, k=None):
        self.ratings = Ratings(ratings)
        if embeddings is None:
            self.embeddings = EmbeddingsFromRatingsIdentity()(self.ratings)
        else:
            self.embeddings = Embeddings(embeddings, norm=True)
        if k is not None:
            self.k_ = k
        self.delete_cache()
        return self

    def set_k(self, k):
        """
        A function to update the
        size :attr:`k_` of the winning committee

        Parameters
        ----------
        k : int
            The new size of the committee.

        Return
        ------
        MultiwinnerRule
            The object itself.
        """
        self.delete_cache()
        self.k_ = k
        return self

    @cached_property
    def winners_(self):
        """
        A function that returns the winners,
        i.e. the members of the elected committee.

        Return
        ------
        int list
            The indexes of the elected candidates.
        """
        raise NotImplementedError
