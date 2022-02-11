
import numpy as np
from embedded_voting.manipulation.coalition.general import ManipulationCoalition
from embedded_voting.scoring.singlewinner.rule_svd_nash import RuleSVDNash
from embedded_voting.scoring.singlewinner.rule_instant_runoff_extension import RuleInstantRunoffExtension
from embedded_voting.scoring.singlewinner.rule_positional_extension_borda import RulePositionalExtensionBorda
from embedded_voting.scoring.singlewinner.rule_positional_extension_k_approval import RulePositionalExtensionKApproval
from embedded_voting.embeddings.embeddings_generator_polarized import EmbeddingsGeneratorPolarized
from embedded_voting import RatingsFromEmbeddingsCorrelated


class ManipulationCoalitionExtension(ManipulationCoalition):
    """
    This class extends the :class:`ManipulationCoalition`
    class to ordinal extension (irv, borda, plurality, etc.), because
    the :class:`ManipulationCoalition` cannot
    be used for ordinal preferences.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    extension : RulePositionalExtension
        The ordinal extension used.
    rule : Rule
        The aggregation rule we want to analysis.

    Attributes
    ----------
    rule : Rule
        The aggregation rule we want to analysis.
    winner_ : int
        The index of the winner of the election without manipulation.
    welfare_ : float list
        The welfares of the candidates without manipulation.
    extended_rule : Rule
        The rule we are analysing
    extension : RulePositionalExtension
        The extension used.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=0.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> extension = RuleInstantRunoffExtension()
    >>> manipulation = ManipulationCoalitionExtension(ratings, embeddings, extension, RuleSVDNash())
    >>> manipulation.winner_
    2
    >>> manipulation.is_manipulable_
    True
    >>> manipulation.worst_welfare_
    0.0
    """

    def __init__(self, ratings, embeddings, extension=None, rule=None):
        super().__init__(ratings, embeddings)
        self.extension = extension
        self.rule = rule
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

    def trivial_manipulation(self, candidate, verbose=False):

        voters_interested = []
        for i in range(self.ratings.n_voters):
            score_i = self.ratings.voter_ratings(i)
            if score_i[self.winner_] < score_i[candidate]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i" %
                  (len(voters_interested), candidate, self.winner_))

        profile = self.ratings.copy()
        for i in voters_interested:
            profile[i][self.winner_] = -1
            profile[i][candidate] = 2

        new_winner = self.extended_rule(profile, self.embeddings).winner_

        if verbose:
            print("Winner is %i" % new_winner)

        return new_winner == candidate


class ManipulationCoalitionBorda(ManipulationCoalitionExtension):
    """
    This class do the coalition manipulation
    analysis for the :class:`RulePositionalExtensionBorda` extension.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    rule : Rule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=0.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> manipulation = ManipulationCoalitionBorda(ratings, embeddings, RuleSVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.is_manipulable_
    False
    >>> manipulation.worst_welfare_
    1.0
    """

    def __init__(self, ratings, embeddings, rule=None):
        if not isinstance(ratings, np.ndarray):
            ratings = ratings.ratings
        super().__init__(ratings, embeddings, extension=RulePositionalExtensionBorda(ratings.shape[1]), rule=rule)


class ManipulationCoalitionKApp(ManipulationCoalitionExtension):
    """
    This class do the coalition manipulation
    analysis for the :class:`RulePositionalExtensionKApproval` extension.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    k : int
        The parameter of the k-approval rule.
    rule : Rule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=0.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> manipulation = ManipulationCoalitionKApp(ratings, embeddings, k=2, rule=RuleSVDNash())
    >>> manipulation.winner_
    1
    >>> manipulation.is_manipulable_
    False
    >>> manipulation.worst_welfare_
    1.0
    """

    def __init__(self, ratings, embeddings, k=2, rule=None):
        if not isinstance(ratings, np.ndarray):
            ratings = ratings.ratings
        super().__init__(ratings, embeddings, extension=RulePositionalExtensionKApproval(ratings.shape[1], k=k), rule=rule)


class ManipulationCoalitionIRV(ManipulationCoalitionExtension):
    """
    This class do the coalition manipulation
    analysis for the :class:`RuleInstantRunoffExtension` extension.

    Parameters
    ----------
    ratings: Ratings or np.ndarray
        The ratings of voters to candidates
    embeddings: Embeddings
        The embeddings of the voters
    rule : Rule
        The aggregation rule we want to analysis.

    Examples
    --------
    >>> np.random.seed(42)
    >>> ratings_dim_candidate = [[1, .2, 0], [.5, .6, .9], [.1, .8, .3]]
    >>> embeddings = EmbeddingsGeneratorPolarized(10, 3)(.8)
    >>> ratings = RatingsFromEmbeddingsCorrelated(coherence=0.8, ratings_dim_candidate=ratings_dim_candidate)(embeddings)
    >>> manipulation = ManipulationCoalitionIRV(ratings, embeddings, RuleSVDNash())
    >>> manipulation.winner_
    2
    >>> manipulation.is_manipulable_
    True
    >>> manipulation.worst_welfare_
    0.0
    """

    def __init__(self, ratings, embeddings, rule=None):
        if not isinstance(ratings, np.ndarray):
            ratings = ratings.ratings
        super().__init__(ratings, embeddings, extension=RuleInstantRunoffExtension(ratings.shape[1]), rule=rule)
