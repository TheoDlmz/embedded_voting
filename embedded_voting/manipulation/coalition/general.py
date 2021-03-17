
import numpy as np
from embedded_voting.utils.cached import DeleteCacheMixin, cached_property
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.svd import SVDNash


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
