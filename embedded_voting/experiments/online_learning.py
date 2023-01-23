import numpy as np
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.rules.singlewinner_rules.rule_sum_ratings import RuleSumRatings
from tqdm import tqdm
import matplotlib.pyplot as plt


class OnlineLearning:
    """
    Class to compare the performance of different aggregators on a given generator.
    
    Parameters
    ----------
    list_agg: list of Aggregator
        List of aggregators to compare.
    generator: TruthGenerator
        Generator to use for the true ratings of the candidates
    
    """

    def __init__(self, list_agg, generator=None):
        self.list_agg = list_agg
        self.generator = generator

    def _run(self, n_candidates=20, n_steps=10):

        results = []
        for agg in self.list_agg:
            agg.reset()

        for i in range(n_steps):
            ratings = self.generator(n_candidates)
            truth = self.generator.ground_truth_
            ratings = np.maximum(ratings, 0)

            # We get the real welfare of each candidate using a SumScoresProfile
            welfare = RuleSumRatings()(Ratings([truth])).welfare_

            results_i = []
            for agg in self.list_agg:
                w = agg(ratings).winner_
                results_i.append(welfare[w])
            results.append(results_i)
        return np.array(results).T

    def __call__(self, n_candidates=20, n_steps=10, n_try=100):
        results = np.zeros((len(self.list_agg), n_steps))
        self.labels_ = [n_candidates*(i+1) for i in range(n_steps)]

        for _ in tqdm(range(n_try)):
            results += self._run(n_candidates, n_steps)

        self.results_ = results / n_try

    def plot(self, show=True):
        rules_names = [agg.name for agg in self.list_agg]
        _ = plt.figure(figsize=(20, 5))
        for i, r in enumerate(self.results_):
            plt.plot(self.labels_, r, 'o-', label=rules_names[i], linewidth=3)

        plt.ylim(0.8, 1)
        plt.xlim(self.labels_[0], self.labels_[-1])
        plt.xlabel("Training test size")
        plt.ylabel("Mean welfare")
        plt.title("Evolution of the welfare with training test size")
        plt.legend()
        plt.grid(0.3)
        if show:
            plt.show()
