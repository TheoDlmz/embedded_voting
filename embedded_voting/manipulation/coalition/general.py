
import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.svd import SVDNash
import matplotlib.pyplot as plt


class ManipulationCoalition(DeleteCacheMixin):
    """
    This general class is used for the analysis of the manipulability of the rule by a coalition of voter.
    It only look if there is a trivial manipulation by a coalition of voter. That means, for some candidate c
    different than the current winner w, gather every voter who prefers c to w, and ask them to put c first and
    w last. If c is the new winner, then the profile can be manipulated.

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
    >>> manipulation = ManipulationCoalition(my_profile, SVDNash())
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

    def trivial_manipulation(self, candidate, verbose=False):
        """
        This function compute if a trivial manipulation is possible for the candidate
        passed as parameter.

        Parameters
        ----------
        candidate : int
            The index of the candidate for which we manipulate.
        verbose : bool
            Verbose mode. By default, is set to False.

        Return
        ------
        bool
            If True, the profile is manipulable for this candidate.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = ManipulationCoalition(my_profile, SVDNash())
        >>> manipulation.trivial_manipulation(0, verbose=True)
        3 voters interested to elect 0 instead of 1
        Winner is 0
        True
        """

        voters_interested = []
        for i in range(self.profile_.n_voters):
            score_i = self.profile_.scores[i]
            if score_i[self.winner_] < score_i[candidate]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i" %
                  (len(voters_interested), candidate, self.winner_))

        old_profile = self.profile_.scores.copy()
        for i in voters_interested:
            self.profile_.scores[i] = np.zeros(self.profile_.n_candidates)
            self.profile_.scores[i][candidate] = 1

        new_winner = self.rule_(self.profile_).winner_
        self.profile_.scores = old_profile

        if verbose:
            print("Winner is %i" % new_winner)

        return new_winner == candidate

    @cached_property
    def is_manipulable_(self):
        """
        A function that quickly compute if the profile is manipulable

        Return
        ------
        bool
            If True, the profile is manipulable for some candidate.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = ManipulationCoalition(my_profile, SVDNash())
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
        A function that compute the worst welfare attainable by coalition manipulation.

        Return
        ------
        float
            The worst welfare.

        Examples
        --------
        >>> np.random.seed(42)
        >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
        >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
        >>> manipulation = ManipulationCoalition(my_profile, SVDNash())
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
        >>> manipulation = ManipulationCoalition(profile, rule=SVDNash())
        >>> maps = manipulation.manipulation_map(profile, map_size=5, show=False)
        >>> maps['worst_welfare']
        array([[0.        , 0.        , 0.        , 0.        , 0.08832411],
               [0.        , 0.        , 0.42649307, 0.18374426, 0.73451954],
               [0.        , 0.        , 0.27448321, 0.34256585, 0.        ],
               [0.        , 0.        , 0.        , 1.        , 0.1774594 ],
               [0.        , 0.        , 0.36950023, 0.69474578, 0.42774779]])
        """

        manipulator_map = np.zeros((map_size, map_size))
        worst_welfare_map = np.zeros((map_size, map_size))

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

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(2, 1, 1)
        ax.imshow(manipulator_map[::-1, ::], vmin=0, vmax=1)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Orthogonality')
        ax.set_title("Proportion of manipulators")
        ax.set_xticks([0, map_size-1], [0, 1])
        ax.set_yticks([0, map_size-1], [1, 0])

        ax = fig.add_subplot(2, 1, 2)
        ax.imshow(worst_welfare_map[::-1, ::], vmin=0, vmax=1)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Orthogonality')
        ax.set_title("Worst welfare")
        ax.set_xticks([0, map_size-1], [0, 1])
        ax.set_yticks([0, map_size-1], [1, 0])

        if show:
            plt.show()  # pragma: no cover

        return {"manipulator": manipulator_map,
                "worst_welfare": worst_welfare_map,
                "fig": fig}
