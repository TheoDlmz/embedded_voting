from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.embeddings.generator import EmbeddingsGeneratorPolarized
from embedded_voting.ratings.ratingsFromEmbeddings import RatingsFromEmbeddingsCorrelated
from embedded_voting.scoring.singlewinner.svd import SVDNash
from embedded_voting.scoring.singlewinner.ordinal import BordaExtension
import numpy as np
import itertools
import matplotlib.pyplot as plt
from embedded_voting.utils.plots import create_map_plot
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings


class SingleVoterManipulation(DeleteCacheMixin):
    """
    This general class is used for the
    analysis of the manipulability of some :class:`ScoringRule`
    by a single voter.

    For instance, what proportion of voters can
    change the result of the rule (to their advantage)
    by giving false preferences ?

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
        The ratings of voters on which we do the analysis.
    rule : ScoringRule
        The aggregation rule we want to analysis.
    winner_ : int
        The index of the winner of the election without manipulation.
    scores_ : float list
        The scores of the candidates without manipulation.
    welfare_ : float list
        The welfares of the candidates without manipulation.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
    >>> manipulation = SingleVoterManipulation(ratings, embeddings, SVDNash())
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
        """
        This function update the ratings of voters
        on which we do the analysis.

        Parameters
        ----------
        ratings : Ratings or np.ndarray
        embeddings : Embeddings

        Return
        ------
        SingleVoterManipulation
            The object itself.
        """
        if embeddings is not None:
            self.embeddings = Embeddings(embeddings)
        self.ratings = Ratings(ratings)
        global_rule = self.rule(self.ratings, self.embeddings)
        self.winner_ = global_rule.winner_
        self.scores_ = global_rule.scores_
        self.welfare_ = global_rule.welfare_
        self.delete_cache()
        return self

    def manipulation_voter(self, i):
        """
        This function return, for the `i^th` voter,
        its favorite candidate that he can turn to
        a winner by manipulating the election.

        Parameters
        ----------
        i : int
            The index of the voter.

        Return
        ------
        int
            The index of the best candidate
            that can be elected by manipulation.
        """
        score_i = self.ratings.voter_ratings(i).copy()
        preferences_order = np.argsort(score_i)[::-1]

        # If the favorite of the voter is the winner, he will not manipulate
        if preferences_order[0] == self.winner_:
            return self.winner_

        n_candidates = self.ratings.n_candidates
        self.ratings[i] = np.ones(n_candidates)
        scores_max = self.rule(self.ratings, self.embeddings).scores_
        self.ratings[i] = np.zeros(n_candidates)
        scores_min = self.rule(self.ratings, self.embeddings).scores_
        self.ratings[i] = score_i

        all_scores = [(s, i, 1) for i, s in enumerate(scores_max)]
        all_scores += [(s, i, 0) for i, s in enumerate(scores_min)]

        all_scores.sort()
        all_scores = all_scores[::-1]

        best_manipulation = np.where(preferences_order == self.winner_)[0][0]
        for (_, i, k) in all_scores:
            if k == 0:
                break

            index_candidate = np.where(preferences_order == i)[0][0]
            if index_candidate < best_manipulation:
                best_manipulation = index_candidate

        best_manipulation = preferences_order[best_manipulation]

        return best_manipulation

    @cached_property
    def manipulation_global_(self):
        """
        This function applies the function
        :meth:`manipulation_voter` to every voter.

        Return
        ------
        int list
            The list of the best candidates that can be
            turned into the winner for each voter.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = SingleVoterManipulation(ratings, embeddings, SVDNash())
        >>> manipulation.manipulation_global_
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 0]
        """
        return [self.manipulation_voter(i) for i in range(self.ratings.n_voters)]

    @cached_property
    def prop_manipulator_(self):
        """
        This function computes the proportion
        of voters that can manipulate the election.

        Return
        ------
        float
            The proportion of voters
            that can manipulate the election.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = SingleVoterManipulation(ratings, embeddings, SVDNash())
        >>> manipulation.prop_manipulator_
        0.4
        """
        return len([x for x in self.manipulation_global_ if x != self.winner_]) / self.ratings.n_voters

    @cached_property
    def avg_welfare_(self):
        """
        The function computes the average welfare
        of the winning candidate after a voter manipulation.

        Return
        ------
        float
            The average welfare.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = SingleVoterManipulation(ratings, embeddings, SVDNash())
        >>> manipulation.avg_welfare_
        0.956...
        """
        return np.mean([self.welfare_[x] for x in self.manipulation_global_])

    @cached_property
    def worst_welfare_(self):
        """
        This function computes the worst possible welfare
        achievable by single voter manipulation.

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
        >>> manipulation = SingleVoterManipulation(ratings, embeddings, SVDNash())
        >>> manipulation.worst_welfare_
        0.891...
        """
        return np.min([self.welfare_[x] for x in self.manipulation_global_])

    @cached_property
    def is_manipulable_(self):
        """
        This function quickly computes
        if the ratings is manipulable or not.

        Return
        ------
        bool
            If True, the ratings is
            manipulable by a single voter.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
        >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
        >>> manipulation = SingleVoterManipulation(ratings, embeddings, SVDNash())
        >>> manipulation.is_manipulable_
        True
        """
        for i in range(self.ratings.n_voters):
            if self.manipulation_voter(i) != self.winner_:
                return True
        return False

    def manipulation_map(self, map_size=20, scores_matrix=None, show=True):
        """
        A function to plot the manipulability
        of the ratings when the ``polarisation`` and the ``coherence``
        of the :class:`ParametricProfile` vary.
        The number of voters, dimensions, and candidates
        are those of the :attr:`profile_`.

        Parameters
        ----------
        map_size : int
            The number of different ``coherence``
            and ``polarisation`` parameters tested.
            The total number of test is `map_size` `^2`.
        scores_matrix : np.ndarray
            Matrix of shape :attr:`~embedded_voting.Profile.embeddings.n_dim`,
            :attr:`~embedded_voting.Profile.n_candidates` containing
            the scores given by each group.
            More precisely, `scores_matrix[i,j]` is the score given by the group
            represented by the dimension `i` to the candidate `j`.
            If None specified, a new matrix is generated for each test.
        show : bool
            If True, display the manipulation maps
            at the end of the function.

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
        >>> manipulation = SingleVoterManipulation(rat, emb, rule=SVDNash())
        >>> maps = manipulation.manipulation_map(map_size=5, show=False)
        >>> maps['manipulator']
        array([[0.02, 0.49, 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.54, 0.  , 0.  , 0.4 , 0.  ],
               [0.01, 0.51, 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.36, 0.  ]])
        """

        manipulator = np.zeros((map_size, map_size))
        worst_welfare = np.zeros((map_size, map_size))
        avg_welfare = np.zeros((map_size, map_size))

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
                manipulator[i, j] = self.prop_manipulator_
                worst_welfare[i, j] = self.worst_welfare_
                avg_welfare[i, j] = self.avg_welfare_

        if show:
            fig = plt.figure(figsize=(15, 5))

            create_map_plot(fig, manipulator, [1, 3, 1], "Proportion of manipulators")
            create_map_plot(fig, avg_welfare, [1, 3, 2], "Average welfare")
            create_map_plot(fig, worst_welfare, [1, 3, 3], "Worst welfare")

            plt.show()

        return {"manipulator": manipulator,
                "worst_welfare": worst_welfare,
                "avg_welfare": avg_welfare}


class SingleVoterManipulationExtension(SingleVoterManipulation):
    """
    This class extends the :class:`SingleVoterManipulation`
    class to ordinal extension (irv, borda, plurality, etc.).

    Parameters
    ----------
    ratings : Profile
        The ratings of voters on which we do the analysis.
    extension : PositionalRuleExtension
        The ordinal extension used.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Attributes
    ----------
    rule : ScoringRule
        The aggregation rule we want to analysis.
    winner_ : int
        The index of the winner of the election without manipulation.
    welfare_ : float list
        The welfares of the candidates without manipulation.
    extended_rule : ScoringRule
        The rule we are analysing
    extension : PositionalRuleExtension
        The extension used.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores_matrix = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(3, 3, scores_matrix)(embeddings, .8)
    >>> extension = BordaExtension(3)
    >>> manipulation = SingleVoterManipulationExtension(ratings, embeddings, extension, SVDNash())
    >>> manipulation.prop_manipulator_
    0.3
    >>> manipulation.manipulation_global_
    [1, 2, 1, 1, 2, 2, 1, 1, 1, 1]
    >>> manipulation.avg_welfare_
    0.7
    """

    def __init__(self, ratings, embeddings, extension, rule=None):
        super().__init__(ratings, embeddings)
        self.rule = rule
        self.extension = extension
        if rule is not None:
            self.extended_rule = self.extension.set_rule(rule)
            self.extended_rule(self.ratings, self.embeddings)
            self.winner_ = self.extended_rule.winner_
            self.welfare_ = self.rule(self.ratings, self.embeddings).welfare_
            self.delete_cache()
        else:
            self.extended_rule = None

    def __call__(self, rule):
        self.rule = rule
        self.extended_rule = self.extension.set_rule(rule)
        self.extended_rule(self.ratings, self.embeddings)
        self.winner_ = self.extended_rule.winner_
        self.welfare_ = self.rule(self.ratings, self.embeddings).welfare_
        self.delete_cache()
        return self

    def manipulation_voter(self, i):
        score_i = self.ratings[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        n_candidates = self.ratings.shape[1]
        points = np.arange(n_candidates)[::-1]
        if preferences_order[0] == self.winner_:
            return self.winner_

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]

        for perm in itertools.permutations(range(n_candidates)):
            self.ratings[i] = points[list(perm)]
            fake_run = self.extended_rule(self.ratings, self.embeddings)
            new_winner = fake_run.winner_
            index_candidate = np.where(preferences_order == new_winner)[0][0]
            if index_candidate < best_manipulation_i:
                best_manipulation_i = index_candidate
                if best_manipulation_i == 0:
                    break

        best_manipulation = preferences_order[best_manipulation_i]
        self.ratings[i] = score_i

        return best_manipulation
