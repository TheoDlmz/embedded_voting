from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.multiwinner.svd import IterSVD
import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_plots():
    np.random.seed(42)
    scores = [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1]]
    probability = [2 / 4, 1 / 4, 1/4]
    my_profile = ParametricProfile(6, 3, 100, scores, probability).set_parameters(1, 1)
    election = IterSVD(my_profile, 3, quota="droop", take_min=True)
    election(my_profile, 4)
    assert election.winners_ == [0, 1, 3, 5]
    election.plot_winners("ternary", show=False)
    election.plot_winners("3D", show=False)
    election.plot_weights("ternary", show=False)
    election.plot_weights("3D", show=False)
    plt.close()

def test_errors():
    scores = [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1]]
    probability = [2 / 4, 1 / 4, 1 / 4]
    my_profile = ParametricProfile(6, 3, 100, scores, probability).set_parameters(1, 1)
    with pytest.raises(ValueError):
        _ = IterSVD(my_profile, 3, quota="toto", take_min=True)
    election = IterSVD(my_profile, 3, quota="droop", take_min=True)
    with pytest.raises(ValueError):
        election.set_quota("toto")
    election.set_quota("classic")
    election.quota = "toto"
    with pytest.raises(ValueError):
        _ = election.winners_
