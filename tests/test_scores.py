from embedded_voting.scoring.singlewinner.features import FeaturesRule
from embedded_voting.scoring.singlewinner.geometric import *
from embedded_voting.scoring.singlewinner.ordinal import *
from embedded_voting.scoring.singlewinner.svd import *
from embedded_voting.scoring.singlewinner.fakesvd import *
from embedded_voting.profile.Profile import Profile
import numpy as np
import pytest


def test_plot():
    my_profile = Profile(5, 3)
    my_profile.uniform_distribution(100)
    election = FeaturesRule()
    election(my_profile)
    election.plot_winner(show=False)
    election.plot_ranking(show=False)
    election.plot_features("3D", show=False)
    election.plot_features("ternary", show=False)
    with pytest.raises(ValueError):
        election.plot_features("toto", show=False)
    with pytest.raises(ValueError):
        election.plot_features("3D", dim=[1, 2, 3, 4], show=False)
    election = SVDMax(my_profile)
    election.plot_features("3D", show=False)
    election.plot_features("ternary", show=False)
    with pytest.raises(ValueError):
        election.plot_features("3D", dim=[1, 2, 3, 4], show=False)
    plt.close()


def test_special_cases():
    my_profile = Profile(3, 3)
    my_profile.add_voters(np.random.rand(10, 3), np.ones((10, 3)))
    election = FeaturesRule(my_profile)
    welfare = election.welfare_
    for s in welfare:
        assert s == 1

    my_profile = Profile(3, 3)
    my_profile.add_voters(np.ones((10, 3)), np.ones((10, 3)))
    election = ZonotopeRule(my_profile)
    scores = election.scores_
    for s in scores:
        assert s == (1, 10)
    election = MaxCubeRule(my_profile)
    scores = election.scores_
    for s in scores:
        assert s == (1, 1)

    my_profile = Profile(3, 10)
    my_profile.add_voters(np.ones((3, 10)), np.random.rand(3, 3))
    election = SVDNash(my_profile, use_rank=True)
    _ = election.ranking_


def test_ordinal():
    my_profile = Profile(3, 3)
    my_profile.add_voters(np.random.rand(10, 3), np.ones((10, 3)))
    with pytest.raises(ValueError):
        PositionalRuleExtension(my_profile, [3, 2, 1, 0], SVDNash())
    election = BordaExtension(my_profile, SVDNash())
    election(my_profile)
    election.plot_fake_profile(show=False)

    election = InstantRunoffExtension(my_profile)
    election(my_profile)


def test_fake_svd():
    my_profile = Profile(3, 3)
    my_profile.add_voters(np.random.rand(10, 3), np.ones((10, 3)))
    election = FakeSVDRule(my_profile, np.dot, use_rank=True)
    election.set_rule(np.prod)
    _ = election.scores_
