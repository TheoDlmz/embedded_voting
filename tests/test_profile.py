from embedded_voting.profile.Profile import Profile
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.profile.MovingVoter import MovingVoterProfile
from embedded_voting.scoring.singlewinner.svd import *
import pytest
import numpy as np
import matplotlib.pyplot as plt


def test_plot():
    my_profile = Profile(5, 3)
    my_profile.uniform_distribution(100)
    my_profile.plot_profile("3D", show=False)
    my_profile.plot_profile("ternary", show=False)
    my_profile.plot_candidates("3D", show=False)
    my_profile.plot_candidates("ternary", show=False)
    my_profile = MovingVoterProfile(SVDNash())
    my_profile.plot_scores_evolution(show=False)
    plt.close()


def test_error():
    my_profile = Profile(5, 3)
    my_profile.uniform_distribution(1)
    with pytest.raises(ValueError):
        my_profile.dilate()
    with pytest.raises(ValueError):
        my_profile.recenter()
    my_profile.uniform_distribution(100)
    with pytest.raises(ValueError):
        my_profile.plot_profile("toto", show=False)
    with pytest.raises(ValueError):
        my_profile.plot_profile("3D", dim=[1, 2, 3, 4], show=False)
    with pytest.raises(ValueError):
        my_profile.plot_candidates("3D", dim=[1, 2, 3, 4], show=False)

    my_profile = ParametricProfile(5, 3, 100)
    with pytest.raises(ValueError):
        my_profile.set_parameters(2, 1)

    new_profile = Profile(5, 3)
    new_profile.reset_profile(my_profile)
    new_profile.reset_profile()
    new_profile = Profile(5, 4)
    with pytest.raises(ValueError):
        new_profile.reset_profile(my_profile)
    new_profile = Profile(6, 3)
    with pytest.raises(ValueError):
        new_profile.reset_profile(my_profile)


def test_particular_case():
    my_profile = Profile(5, 2)
    for i in range(3):
        my_profile.add_voter([.5, .5], np.random.rand(5))
    my_profile.dilate()
    for i in range(3):
        assert list(my_profile.embeddings[i]) == [0.7071067811865475, 0.7071067811865475]
    my_profile.recenter()
    for i in range(3):
        assert list(my_profile.embeddings[i]) == [0.7071067811865475, 0.7071067811865475]

    my_profile = Profile(5, 2)
    for i in range(3):
        my_profile.add_voter([-.5, -.5], np.random.rand(5))
    my_profile.recenter()
    for i in range(3):
        assert list(my_profile.embeddings[i]) == [0.7071067811865475, 0.7071067811865475]


def test_rotations():
    my_profile = Profile(5, 2)
    for i in range(20):
        my_profile.add_voter([0.6, 0.4], np.random.rand(5))
    for i in range(3):
        my_profile.add_voter([0.4, 0.6], np.random.rand(5))
    my_profile.dilate()
    for i in range(5):
        assert round(list(my_profile.embeddings[i])[0], 2) == 0.87

    my_profile = Profile(5, 2)
    for i in range(20):
        my_profile.add_voter([0.6, 0.4], np.random.rand(5))
    for i in range(3):
        my_profile.add_voter([0.4, 0.6], np.random.rand(5))
    my_profile.dilate(approx=False)
    for i in range(5):
        assert round(list(my_profile.embeddings[i])[0], 2) == 1

    my_profile = Profile(5, 2)
    for i in range(20):
        my_profile.add_voter([1, 0], np.random.rand(5))
    for i in range(3):
        my_profile.add_voter([0.8, 0.2], np.random.rand(5))
    my_profile.recenter(approx=False)
    for i in range(5):
        assert round(list(my_profile.embeddings[i])[0], 2) == 0.79

