from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.multiwinner.svd import IterSVD
import numpy as np


def test_plots():
    np.random.seed(42)
    scores = [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1]]
    probability = [2 / 4, 1 / 4, 1/4]
    my_profile = ParametricProfile(6, 3, 100, scores, probability).set_parameters(1, 1)
    election = IterSVD(my_profile, 4, quota="droop", take_min=True)
    election(my_profile)
    assert election.winners_ == [0, 3, 5, 1]
    election.plot_winners("ternary", show=False)
    election.plot_winners("3D", show=False)
    election.plot_weights("ternary", show=False)
    election.plot_weights("3D", show=False)
