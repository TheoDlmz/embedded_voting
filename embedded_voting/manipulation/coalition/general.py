
import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.profile.parametric import ProfileGenerator
from embedded_voting.scoring.singlewinner.svd import SVDNash
import matplotlib.pyplot as plt
from embedded_voting.utils.plots import create_map_plot


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
    the profile can be manipulated.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Attributes
    ----------
    profile_ : Profile
        The profile of voter on which we do the analysis.
    rule_ : ScoringRule
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
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> profile = ProfileGenerator(10, 3, 3, scores)(0.8, 0.8)
    >>> manipulation = ManipulationCoalition(profile, SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.welfare_
    [0.8914711297748728, 1.0, 0.0]

    """
    def __init__(self, profile, rule=None):
        self.profile_ = profile
        self.rule_ = rule
        if rule is not None:
            global_rule = self.rule_(self.profile_)
            self.winner_ = global_rule.winner_
            self.scores_ = global_rule.scores_
            self.welfare_ = global_rule.welfare_
        else:
            self.winner_ = None
            self.scores_ = None
            self.welfare_ = None

    def __call__(self, rule):
        self.rule_ = rule
        global_rule = self.rule_(self.profile_)
        self.winner_ = global_rule.winner_
        self.scores_ = global_rule.scores_
        self.welfare_ = global_rule.welfare_
        self.delete_cache()
        return self

    def set_profile(self, profile):
        self.profile_ = profile
        global_rule = self.rule_(self.profile_)
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
            If True, the profile is manipulable
            for this candidate.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> profile = ProfileGenerator(10, 3, 3, scores)(0.8, 0.8)
        >>> manipulation = ManipulationCoalition(profile, SVDNash())
        >>> manipulation.trivial_manipulation(0, verbose=True)
        4 voters interested to elect 0 instead of 1
        Winner is 0
        True
        """

        voters_interested = []
        for i in range(self.profile_.n_voters):
            score_i = self.profile_.ratings[i]
            if score_i[self.winner_] < score_i[candidate]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i" %
                  (len(voters_interested), candidate, self.winner_))

        old_profile = self.profile_.ratings.copy()
        for i in voters_interested:
            self.profile_.ratings[i] = np.zeros(self.profile_.n_candidates)
            self.profile_.ratings[i][candidate] = 1

        new_winner = self.rule_(self.profile_).winner_
        self.profile_.ratings = old_profile

        if verbose:
            print("Winner is %i" % new_winner)

        return new_winner == candidate

    @cached_property
    def is_manipulable_(self):
        """
        A function that quickly computes
        if the profile is manipulable.

        Return
        ------
        bool
            If True, the profile is
            manipulable for some candidate.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> profile = ProfileGenerator(10, 3, 3, scores)(0.8, 0.8)
        >>> manipulation = ManipulationCoalition(profile, SVDNash())
        >>> manipulation.is_manipulable_
        True
        """

        for i in range(self.profile_.n_candidates):
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
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> profile = ProfileGenerator(10, 3, 3, scores)(0.8, 0.8)
        >>> manipulation = ManipulationCoalition(profile, SVDNash())
        >>> manipulation.worst_welfare_
        0.0
        """
        worst_welfare = self.welfare_[self.winner_]
        for i in range(self.profile_.n_candidates):
            if i == self.winner_:
                continue
            if self.trivial_manipulation(i):
                worst_welfare = min(worst_welfare, self.welfare_[i])
        return worst_welfare

    def manipulation_map(self, map_size=20, scores_matrix=None, show=True):
        """
        A function to plot the manipulability
        of the profile when the
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
        >>> profile = ProfileGenerator(100, 5, 3)(0, 0)
        >>> manipulation = ManipulationCoalition(profile, rule=SVDNash())
        >>> maps = manipulation.manipulation_map(map_size=5, show=False)
        >>> maps['worst_welfare']
        array([[0.        , 0.        , 0.41741546, 1.        , 0.4982297 ],
               [0.        , 0.        , 0.        , 0.14175864, 1.        ],
               [0.        , 0.        , 0.5536796 , 0.43986763, 0.47120528],
               [0.        , 0.        , 0.        , 0.28219718, 0.03216955],
               [0.        , 0.        , 0.        , 0.63924358, 0.28677042]])
        """

        manipulator_map = np.zeros((map_size, map_size))
        worst_welfare_map = np.zeros((map_size, map_size))

        n_candidates = self.profile_.n_candidates
        n_voters = self.profile_.n_voters
        n_dim = self.profile_.embeddings.n_dim

        generator = ProfileGenerator(n_voters, n_candidates, n_dim)

        if scores_matrix is not None:
            generator.set_scores(scores_matrix)

        for i in range(map_size):
            for j in range(map_size):
                if scores_matrix is None:
                    generator.set_scores()
                self.set_profile(generator(i / (map_size-1), j / (map_size-1)))

                worst_welfare = self.welfare_[self.winner_]
                is_manipulable = 0
                for candidate in range(self.profile_.n_candidates):
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
