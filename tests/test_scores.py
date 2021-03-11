from embedded_voting.scoring.singlewinner.features import FeaturesRule
from embedded_voting.scoring.singlewinner.svd import SVDNash
from embedded_voting.scoring.singlewinner.geometric import *
from embedded_voting.profile.Profile import Profile
import numpy as np


def test_plot():
    my_profile = Profile(5, 3)
    my_profile.uniform_distribution(100)
    election = FeaturesRule()
    election(my_profile)
    election.plot_winner(show=False)
    election.plot_ranking(show=False)
    election.plot_features(show=False)


def special_case():
    my_profile = Profile(3, 3)
    my_profile.add_voters(np.random.rand(10, 3), np.ones(10, 3))
    election = FeaturesRule(my_profile)
    assert election.welfare_ == np.ones(3)
    _ = SVDNash(my_profile, use_rank=True)

    my_profile = Profile(3, 3)
    my_profile.add_voters(np.ones(10, 3), np.ones(10, 3))
    election = ZonotopeRule(my_profile)
    assert election.scores_ == [(1, 1)]*10
    election = MaxCubeRule(my_profile)
    assert election.scores_ == [(1, 1)]*10


