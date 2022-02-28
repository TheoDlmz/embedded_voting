from embedded_voting.rules.singlewinner_rules.rule_features import RuleFeatures
from embedded_voting.rules.singlewinner_rules.rule_svd_max import RuleSVDMax
from embedded_voting.rules.multiwinner_rules.multiwinner_rule_iter_svd import MultiwinnerRuleIterSVD
from embedded_voting.ratings.ratings import Ratings
from embedded_voting.embeddings.embeddings import Embeddings
import numpy as np
import pytest


def test_features():
    ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    embeddings = Embeddings(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]]), norm=True)
    election = RuleFeatures()(ratings, embeddings)
    election.plot_features("3D", show=False)
    election.plot_features("ternary", show=False)
    with pytest.raises(ValueError):
        election.plot_features("3D", dim=[0, 1], show=False)


def test_svd():
    ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    embeddings = Embeddings(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]]), norm=True)
    election = RuleSVDMax()(ratings, embeddings)
    election.plot_features("3D", show=False)
    election.plot_features("ternary", show=False)
    with pytest.raises(ValueError):
        election.plot_features("3D", dim=[0, 1], show=False)

    embeddings = Embeddings(np.array(([[1, 1, 2], [0, 1, 1]])), norm=True)
    RuleSVDMax()(ratings, embeddings)


def test_multi():
    ratings = Ratings(np.array([[.5, .6, .3], [.7, 0, .2], [.2, 1, .8]]))
    embeddings = Embeddings(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]]), norm=True)
    election = MultiwinnerRuleIterSVD(k=2)(ratings, embeddings)
    election.plot_weights("3D", show=False)
    election.plot_weights("ternary", show=False)
    with pytest.raises(ValueError):
        election.plot_weights("3D", dim=[0, 1], show=False)

    election.plot_winners("3D", show=False)
