from embedded_voting.profile.Profile import Profile
import pytest


def test_plot():
    my_profile = Profile(5, 3)
    my_profile.uniform_distribution(100)
    my_profile.plot_profile("3D", show=False)
    my_profile.plot_profile("ternary", show=False)
    my_profile.plot_candidates("3D", show=False)
    my_profile.plot_candidates("ternary", show=False)


def test_error():
    my_profile = Profile(5, 3)
    my_profile.uniform_distribution(1)
    with pytest.raises(ValueError):
        my_profile.dilate_profile()
