from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.svd import SVDNash
from embedded_voting.scoring.singlewinner.ordinal import BordaExtension
import numpy as np
import itertools
import matplotlib.pyplot as plt


class SingleVoterManipulation(DeleteCacheMixin):
    """
    This general class is used for the analysis of the manipulability of the rule by a single voter.
    For instance, what proportion of voter can change the result of the rule (to their advantage)
    by giving fake preferences ?

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we do the analysis
    rule : ScoringRule
        The rule we are analysing

    Attributes
    ----------
    profile_ : Profile
        The profile of voter on which we do the analysis
    rule_ : ScoringRule
        The rule we are analysing
    winner_ : int
        The index of the winner of the election without manipulation
    scores_ : float list
        The scores of the candidates without manipulation
    welfare_ : float list
        The welfare of the candidates without manipulation

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> manipulation = SingleVoterManipulation(my_profile, SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.welfare_
    [0.700152659355562, 1.0, 0.0]
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

    def manipulation_voter(self, i):
        """
        This function return, for each voter, its favorite candidate that he can turn to a winner by
        manipulating the election.

        Parameters
        ----------
        i : int
            The index of the voter.

        Return
        ------
        int
            The index of the candidate that can be elected by manipulation.
        """
        score_i = self.profile_.scores[i].copy()
        preferences_order = np.argsort(score_i)[::-1]

        # If the favorite of the voter is the winner, he will not manipulate
        if preferences_order[0] == self.winner_:
            return self.winner_

        self.profile_.scores[i] = np.ones(self.profile_.n_candidates)
        scores_max = self.rule_(self.profile_).scores_
        self.profile_.scores[i] = np.zeros(self.profile_.n_candidates)
        scores_min = self.rule_(self.profile_).scores_
        self.profile_.scores[i] = score_i

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
        This function apply the function manipulation_voter to every voter

        Return
        ------
        int list
            The list of the best candidates that can be turned into the winner for each voter.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = SingleVoterManipulation(my_profile, SVDNash())
        >>> manipulation.manipulation_global_
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        """
        return [self.manipulation_voter(i) for i in range(self.profile_.n_voters)]

    @cached_property
    def prop_manipulator_(self):
        """
        This function compute the proportion of voters that can manipulate in the population.

        Return
        ------
        float
            The proportion of voters that can manipulate in the population.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = SingleVoterManipulation(my_profile, SVDNash())
        >>> manipulation.prop_manipulator_
        0.3
        """
        return len([x for x in self.manipulation_global_ if x != self.winner_]) / self.profile_.n_voters

    @cached_property
    def avg_welfare_(self):
        """
        The function compute the average welfare of the candidate selected by each voter.

        Return
        ------
        float
            The average welfare.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = SingleVoterManipulation(my_profile, SVDNash())
        >>> manipulation.avg_welfare_
        0.9100457978066686
        """
        return np.mean([self.welfare_[x] for x in self.manipulation_global_])

    @cached_property
    def worst_welfare_(self):
        """
        This function compute the worst possible welfare achievable by manipulation.

        Return
        ------
        float
            The worst welfare.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = SingleVoterManipulation(my_profile, SVDNash())
        >>> manipulation.worst_welfare_
        0.700152659355562
        """
        return np.min([self.welfare_[x] for x in self.manipulation_global_])

    @cached_property
    def is_manipulable_(self):
        """
        This function quickly compute if the profile is manipulable or not.

        Return
        ------
        bool
            If True, the profile is manipulable by a single voter.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = SingleVoterManipulation(my_profile, SVDNash())
        >>> manipulation.is_manipulable_
        True
        """
        for i in range(self.profile_.n_voters):
            if self.manipulation_voter(i) != self.winner_:
                return True
        return False

    def manipulation_map(self, parametric_profile, map_size=20, scores_matrix=None, show=True):
        """
        A function to plot the manipulability of the profile when the polarisation and the coherence vary.

        Parameters
        ----------
        parametric_profile : ParametricProfile
            The profile on which we do the manipulation. The only needed information are : :attr:`n_voters`,
            :attr:`n_candidates`, :attr:`n_dim` and :attr:`prob`.
        map_size : int
            The number of different coherence and polarisation parameters tested.
            The total number of test is :attr:`map_size`^2.
        scores_matrix : np.ndarray
            Matrix of shape :attr:`n_dim`, :attr:`n_candidates` containing the scores given by
            each group. More precisely, `scores_matrix[i,j]` is the score given by the group
            represented by the dimension i to the candidate j.
            If None specified, a new matrix is generated for each test.
        show : bool
            If True, display the manipulation maps at the end of the function

        Return
        ------
        dict
            The manipulation maps : `manipulator` for the proportion of manipulator, `worst_welfare` and `avg_welfare`
            for the welfare. `fig` contains the matplotlib figure with the plots.

        Examples
        --------
        >>> np.random.seed(42)
        >>> profile = ParametricProfile(5, 3, 100)
        >>> manipulation = SingleVoterManipulation(profile, rule=SVDNash())
        >>> maps = manipulation.manipulation_map(profile, map_size=5, show=False)
        >>> maps['manipulator']
        array([[0.  , 0.  , 0.  , 0.13, 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.25, 0.  , 0.  , 0.43, 0.  ],
               [0.  , 0.53, 0.5 , 0.  , 0.  ],
               [0.  , 0.14, 0.  , 0.  , 0.38]])
        """

        manipulator = np.zeros((map_size, map_size))
        worst_welfare = np.zeros((map_size, map_size))
        avg_welfare = np.zeros((map_size, map_size))
        if scores_matrix is not None:
            parametric_profile.set_scores(scores_matrix)

        n_candidates = parametric_profile.n_candidates
        n_dim = parametric_profile.n_dim
        for i in range(map_size):
            for j in range(map_size):
                if scores_matrix is None:
                    new_scores = np.random.rand(n_dim, n_candidates)
                    parametric_profile.set_scores(new_scores)
                parametric_profile.set_parameters(i / (map_size-1), j / (map_size-1))
                self.set_profile(parametric_profile)
                manipulator[i, j] = self.prop_manipulator_
                worst_welfare[i, j] = self.worst_welfare_
                avg_welfare[i, j] = self.avg_welfare_

        fig = plt.figure(figsize=(15, 5))

        ax = fig.add_subplot(3, 1, 1)
        ax.imshow(manipulator[::-1, ::], vmin=0, vmax=1)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Orthogonality')
        ax.set_title("Proportion of manipulators")
        ax.set_xticks([0, map_size-1], [0, 1])
        ax.set_yticks([0, map_size-1], [1, 0])

        ax = fig.add_subplot(3, 1, 2)
        ax.imshow(avg_welfare[::-1, ::], vmin=0, vmax=1)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Orthogonality')
        ax.set_title("Average welfare")
        ax.set_xticks([0, map_size-1], [0, 1])
        ax.set_yticks([0, map_size-1], [1, 0])

        ax = fig.add_subplot(3, 1, 3)
        ax.imshow(worst_welfare[::-1, ::], vmin=0, vmax=1)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Orthogonality')
        ax.set_title("Worst welfare")
        ax.set_xticks([0, map_size-1], [0, 1])
        ax.set_yticks([0, map_size-1], [1, 0])

        if show:
            plt.show()  # pragma: no cover

        return {"manipulator": manipulator,
                "worst_welfare": worst_welfare,
                "avg_welfare": avg_welfare,
                "fig": fig}


class SingleVoterManipulationExtension(SingleVoterManipulation):
    """
    This class extends the SingleVoterManipulation class to ordinal extension (irv, borda, plurality, etc.).

    Parameters
    ----------
    profile : Profile
        The profile of voter on which we do the analysis.
    extension : PositionalRuleExtension
        The extension used.
    rule : ScoringRule
        The rule we are analysing.

    Attributes
    ----------
    rule_ : ScoringRule
        The rule we are analysing
    extended_rule : ScoringRule
        The rule we are analysing
    extension : PositionalRuleExtension
        The extension used.
    winner_ : int
        The index of the winner of the election without manipulation
    welfare_ : float list
        The welfare of the candidates without manipulation

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> extension = BordaExtension(my_profile)
    >>> manipulation = SingleVoterManipulationExtension(my_profile, extension, SVDNash())
    >>> manipulation.prop_manipulator_
    0.1
    >>> manipulation.manipulation_global_
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> manipulation.avg_welfare_
    0.9
    """

    def __init__(self, profile, extension, rule=None):
        super().__init__(profile)
        self.rule_ = rule
        self.extension = extension
        if rule is not None:
            self.extended_rule = self.extension.set_rule(rule)
            self.extended_rule(self.profile_)
            self.winner_ = self.extended_rule.winner_
            self.welfare_ = self.rule_(self.profile_).welfare_
            self.delete_cache()
        else:
            self.extended_rule = None

    def __call__(self, rule):
        self.rule_ = rule
        self.extended_rule = self.extension.set_rule(rule)
        self.extended_rule(self.profile_)
        self.winner_ = self.extended_rule.winner_
        self.welfare_ = self.rule_(self.profile_).welfare_
        self.delete_cache()
        return self

    def manipulation_voter(self, i):
        score_i = self.profile_.scores[i].copy()
        preferences_order = np.argsort(score_i)[::-1]
        points = np.arange(self.profile_.n_candidates)[::-1]
        if preferences_order[0] == self.winner_:
            return self.winner_

        best_manipulation_i = np.where(preferences_order == self.winner_)[0][0]

        for perm in itertools.permutations(range(self.profile_.n_candidates)):
            self.profile_.scores[i] = points[list(perm)]
            fake_run = self.extended_rule(self.profile_)
            new_winner = fake_run.winner_
            index_candidate = np.where(preferences_order == new_winner)[0][0]
            if index_candidate < best_manipulation_i:
                best_manipulation_i = index_candidate
                if best_manipulation_i == 0:
                    break

        best_manipulation = preferences_order[best_manipulation_i]
        self.profile_.scores[i] = score_i

        return best_manipulation
