from embedded_voting.scoring.singlewinner.fast import FastNash
from embedded_voting.profile.fastprofile import FastProfile


class Aggregator:
    """
    A class for an election generator with memory.
    You can run an election by calling
    it with the matrix of scores.

    Parameters
    ----------
    rule: ScoringRule
        The aggregation rule you want to use in your elections. Default is :class:`~embedded_voting.FastNash`

    Attributes
    ----------
    embedder: FastProfile
        The profile that keeps in memory information about the voters.

    rule: ScoringRule
        The scoring rule used for the elections.

    Examples
    --------
    >>> aggregator = Aggregator()
    >>> results = aggregator([[7, 5, 9, 5, 1, 8], [7, 5, 9, 5, 2, 7], [6, 4, 2, 4, 4, 6], [3, 8, 1, 3, 7, 8]])
    >>> results.ranking_
    [5, 0, 1, 3, 2, 4]
    >>> results.winner_
    5
    >>> results = aggregator([[2, 4, 8], [9, 2, 1], [0, 2, 5], [4, 5, 3]], train=True)
    >>> results.ranking_
    [2, 1, 0]
    """

    def __init__(self, rule=None):
        if rule is None:
            rule = FastNash()
        self.rule = rule
        self.embedder = None

    def __call__(self, scores, train=False):
        """
        This function run an election using the :attr:`embedder` and the scores.

        Parameters
        ----------
        scores: np.ndarray or list
            The matrix of scores given by the voters. ``scores[i,j]`` corresponds to the
            score given by the voter i to candidate j.

        train: bool
            If True, we retrain the :attr:`embedder` before doing the election (using the
            data of the election).
        """
        if self.embedder is None:
            self.embedder = FastProfile(scores)
        else:
            self.embedder(scores)

        if train:
            self.retrain()

        self.rule.delete_cache()

        return self.rule(self.embedder)

    def retrain(self):
        """
        This function can be used to train the embedder on the newest data
        it gathered during the recent elections.

        Return
        ------
        Aggregator
            The object
        """
        self.embedder.train()
        return self
