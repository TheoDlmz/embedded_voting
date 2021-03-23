
import numpy as np
from embedded_voting.manipulation.coalition.general import ManipulationCoalition
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.svd import SVDNash
from embedded_voting.scoring.singlewinner.ordinal import InstantRunoffExtension, BordaExtension, KApprovalExtension


class ManipulationCoalitionExtension(ManipulationCoalition):
    """
    This class extends the :class:`ManipulationCoalition`
    class to ordinal extension (irv, borda, plurality, etc.), because
    the :class:`ManipulationCoalition` cannot
    be used for ordinal preferences.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    extension : PositionalRuleExtension
        The ordinal extension used.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Attributes
    ----------
    rule_ : ScoringRule
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
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> extension = InstantRunoffExtension(my_profile)
    >>> manipulation = ManipulationCoalitionExtension(my_profile, extension, SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.is_manipulable_
    False
    >>> manipulation.worst_welfare_
    1.0
    """

    def __init__(self, profile, extension=None, rule=None):
        super().__init__(profile)
        self.extension = extension
        self.rule_ = rule
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

    def trivial_manipulation(self, candidate, verbose=False):

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
            self.profile_.scores[i][self.winner_] = -1
            self.profile_.scores[i][candidate] = 2

        new_winner = self.extended_rule(self.profile_).winner_
        self.profile_.scores = old_profile

        if verbose:
            print("Winner is %i" % new_winner)

        return new_winner == candidate


class ManipulationCoalitionBorda(ManipulationCoalitionExtension):
    """
    This class do the coalition manipulation
    analysis for the :class:`BordaExtension` extension.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> manipulation = ManipulationCoalitionBorda(my_profile, rule=SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.is_manipulable_
    True
    >>> manipulation.worst_welfare_
    0.0
    """

    def __init__(self, profile, rule=None):
        super().__init__(profile, extension=BordaExtension(profile), rule=rule)


class ManipulationCoalitionKApp(ManipulationCoalitionExtension):
    """
    This class do the coalition manipulation
    analysis for the :class:`KApprovalExtension` extension.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    k : int
        The parameter of the k-approval rule.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> manipulation = ManipulationCoalitionKApp(my_profile, k=2, rule=SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.is_manipulable_
    True
    >>> manipulation.worst_welfare_
    0.0
    """

    def __init__(self, profile, k=2, rule=None):
        super().__init__(profile, extension=KApprovalExtension(profile, k=k), rule=rule)


class ManipulationCoalitionIRV(ManipulationCoalitionExtension):
    """
    This class do the coalition manipulation
    analysis for the :class:`InstantRunoffExtension` extension.

    Parameters
    ----------
    profile : Profile
        The profile of voters on which we do the analysis.
    rule : ScoringRule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> scores = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> my_profile = ParametricProfile(3, 3, 10, scores).set_parameters(0.8, 0.8)
    >>> manipulation = ManipulationCoalitionIRV(my_profile, SVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.is_manipulable_
    False
    >>> manipulation.worst_welfare_
    1.0
    """

    def __init__(self, profile, rule=None):
        super().__init__(profile, extension=InstantRunoffExtension(profile), rule=rule)
