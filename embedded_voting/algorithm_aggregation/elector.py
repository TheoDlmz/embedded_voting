from embedded_voting.algorithm_aggregation.embedder import Embedder
from embedded_voting.scoring.singlewinner.fast import FastNash


class Elector:
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
    embedder: Embedder
        The embedder that keeps in memory information about the voters.

    rule: ScoringRule
        The scoring rule used for the elections.

    Examples
    --------
    >>> my_elector = Elector()
    >>> results = my_elector([[7, 5, 9, 6, 1], [7, 5, 9, 5, 2], [6, 4, 2, 4, 4], [3, 8, 1, 7, 8]])
    >>> results.ranking_
    [1, 3, 0, 4, 2]
    >>> results.winner_
    1
    >>> results = my_elector([[2, 4, 8], [9, 2, 1], [0, 2, 5], [4, 5, 3]], train=True)
    >>> results.ranking_
    [2, 0, 1]
    """

    def __init__(self, rule=FastNash()):
        self.embedder = None
        self.rule = rule

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
            self.embedder = Embedder(scores)
        else:
            self.embedder(scores)

        self.retrain()
        self.rule.delete_cache()

        return self.rule(self.embedder)

    def retrain(self):
        """
        This function can be used to train the embedder on the newest data
        it gathered during the recent elections.

        Return
        ------
        Elector
            The object
        """
        self.embedder.train()
        return self

    def update_rule(self, rule):
        """
        This function can be used to change the aggregation rule used during
        the elections.

        Parameters
        ----------
        rule: ScoringRule
            The new aggregation rule to use.

        Return
        ------
        Elector
            The object

        Examples
        --------
        >>> my_elector = Elector()
        >>> my_elector.update_rule(FastNash())
        <embedded_voting.algorithm_aggregation.elector.Elector object at ...>
        """
        self.rule = rule
        return self
