
import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.scoring.singlewinner.svd import SVDNash
import matplotlib.pyplot as plt
from embedded_voting.utils.plots import create_map_plot
from embedded_voting.embeddings.generator import EmbeddingsGeneratorPolarized
from embedded_voting.ratings.ratingsFromEmbeddings import RatingsFromEmbeddingsCorrelated
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


class ManipulationCoalition(DeleteCacheMixin):
    """
    This general class is used for the analysis of
    the manipulability of the rule by a coalition of voter.

    It only look if there is a trivial
    manipulation by a coalition of voter.
    That means, for some candidate `c`
    different than the current winner `w`,
    gather every voter who prefers `c` to `w`,
    and ask them to put `c` first and `w` last.
    If `c` is the new winner, then
    the ratings can be manipulated.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Attributes
    ----------
    ratings : Profile
        The ratings of voter on which we do the analysis.
    rule : ScoringRule
        The aggregation rule we want to analysis.
    winner_ : int
        The index of the winner of the election without manipulation.
    scores_ : float list
        The scores of the candidates without manipulation.
    welfare_ : float list
        The welfare of the candidates without manipulation.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
    >>> manipulation = ManipulationCoalition(ratings, embeddings, SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.welfare_
    [0.89..., 1.0, 0.0]

    """
    def __init__(self, ratings, embeddings, rule=None):
        self.ratings = Ratings(ratings)
        self.embeddings = Embeddings(embeddings)
        self.rule = rule
        if rule is not None:
            global_rule = self.rule(self.ratings, self.embeddings)
            self.winner_ = global_rule.winner_
            self.scores_ = global_rule.scores_
            self.welfare_ = global_rule.welfare_
        else:
            self.winner_ = None
            self.scores_ = None
            self.welfare_ = None

    def __call__(self, rule):
        self.rule = rule
        global_rule = self.rule(self.ratings, self.embeddings)
        self.winner_ = global_rule.winner_
        self.scores_ = global_rule.scores_
        self.welfare_ = global_rule.welfare_
        self.delete_cache()
        return self

    def set_profile(self, ratings, embeddings=None):
        if embeddings is not None:
            self.embeddings = Embeddings(embeddings)
        self.ratings = Ratings(ratings)
        global_rule = self.rule(self.ratings, self.embeddings)
        self.winner_ = global_rule.winner_
        self.scores_ = global_rule.scores_
        self.welfare_ = global_rule.welfare_
        self.delete_cache()
        return self

    def trivial_manipulation(self, candidate, verbose=False):
        """
        This function computes if
        a trivial manipulation is
        possible for the candidate
        passed as parameter.

        Parameters
        ----------
        candidate : int
            The index of the candidate
            for which we manipulate.
        verbose : bool
            Verbose mode.
            By default, is set to False.

        Return
        ------
        bool
            If True, the ratings is manipulable
            for this candidate.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = ManipulationCoalition(ratings, embeddings, SVDNash())
        >>> manipulation.trivial_manipulation(0, verbose=True)
        2 voters interested to elect 0 instead of 1
        Winner is 0
        True
        """

        voters_interested = []
        for i in range(self.ratings.shape[1]):
            score_i = self.ratings[i]
            if score_i[self.winner_] < score_i[candidate]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i" %
                  (len(voters_interested), candidate, self.winner_))

        profile = self.ratings.copy()
        for i in voters_interested:
            profile[i] = np.zeros(self.ratings.shape[1])
            profile[i][candidate] = 1

        new_winner = self.rule(profile, self.embeddings).winner_

        if verbose:
            print("Winner is %i" % new_winner)

        return new_winner == candidate

    @cached_property
    def is_manipulable_(self):
        """
        A function that quickly computes
        if the ratings is manipulable.

        Return
        ------
        bool
            If True, the ratings is
            manipulable for some candidate.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = ManipulationCoalition(ratings, embeddings, SVDNash())
        >>> manipulation.is_manipulable_
        True
        """

        for i in range(self.ratings.n_candidates):
            if i == self.winner_:
                continue
            if self.trivial_manipulation(i):
                return True
        return False

    @cached_property
    def worst_welfare_(self):
        """
        A function that compute the worst
        welfare attainable by coalition manipulation.

        Return
        ------
        float
            The worst welfare.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = ManipulationCoalition(ratings, embeddings, SVDNash())
        >>> manipulation.worst_welfare_
        0.0
        """
        worst_welfare = self.welfare_[self.winner_]
        for i in range(self.ratings.n_candidates):
            if i == self.winner_:
                continue
            if self.trivial_manipulation(i):
                worst_welfare = min(worst_welfare, self.welfare_[i])
        return worst_welfare

    def manipulation_map(self, map_size=20, scores_matrix=None, show=True):
        """
        A function to plot the manipulability
        of the ratings when the
        ``polarisation`` and the ``coherence`` vary.

        Parameters
        ----------
        map_size : int
            The number of different coherence and polarisation parameters tested.
            The total number of test is `map_size` ^2.
        scores_matrix : np.ndarray
            Matrix of shape :attr:`~embedded_voting.Profile.n_dim`,
            :attr:`~embedded_voting.Profile.n_candidates`
            containing the scores given by each group.
            More precisely, `scores_matrix[i,j]` is the
            score given by the group represented by
            the dimension `i` to the candidate `j`.
            If not specified, a new matrix is
            generated for each test.
        show : bool
            If True, displays the manipulation
            maps at the end of the function.

        Return
        ------
        dict
            The manipulation maps :
            ``manipulator`` for the proportion of manipulator,
            ``worst_welfare`` and ``avg_welfare``
            for the welfare maps.

        Examples
        --------
        >>> np.random.seed(42)
        >>> emb = EmbeddingsGeneratorPolarized(100, 3)(0)
        >>> rat = RatingsFromEmbeddingsCorrelated(5, 3)(emb)
        >>> manipulation = ManipulationCoalition(rat, emb, SVDNash())
        >>> maps = manipulation.manipulation_map(map_size=5, show=False)
        >>> maps['worst_welfare']
        array([[1.        , 0.98024051, 1.        , 1.        , 1.        ],
               [1.        , 1.        , 1.        , 1.        , 1.        ],
               [0.87833419, 1.        , 1.        , 0.97819913, 1.        ],
               [0.80173984, 0.74016286, 1.        , 1.        , 1.        ],
               [0.81466796, 1.        , 1.        , 0.92431457, 1.        ]])
        """

        manipulator_map = np.zeros((map_size, map_size))
        worst_welfare_map = np.zeros((map_size, map_size))

        n_voters, n_candidates = self.ratings.shape
        n_dim = self.embeddings.n_dim

        embeddings_generator = EmbeddingsGeneratorPolarized(n_voters, n_dim)
        ratings_generator = RatingsFromEmbeddingsCorrelated(n_candidates, n_dim)

        if scores_matrix is not None:
            ratings_generator.set_scores(scores_matrix)
        for i in range(map_size):
            for j in range(map_size):
                if scores_matrix is None:
                    ratings_generator.set_scores()

                embeddings = embeddings_generator(polarisation=i/(map_size-1))
                ratings = ratings_generator(embeddings, coherence=j/(map_size-1))
                self.set_profile(ratings, embeddings)

                worst_welfare = self.welfare_[self.winner_]
                is_manipulable = 0
                for candidate in range(self.ratings.n_candidates):
                    if candidate == self.winner_:
                        continue
                    if self.trivial_manipulation(candidate):
                        is_manipulable = 1
                        worst_welfare = min(worst_welfare, self.welfare_[candidate])

                manipulator_map[i, j] = is_manipulable
                worst_welfare_map[i, j] = worst_welfare

        if show:
            fig = plt.figure(figsize=(10, 5))

            create_map_plot(fig, manipulator_map, [1, 2, 1], "Proportion of manipulators")
            create_map_plot(fig, worst_welfare_map, [1, 2, 2], "Worst welfare")

            plt.show()

        return {"manipulator": manipulator_map,
                "worst_welfare": worst_welfare_map}
