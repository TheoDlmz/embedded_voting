
from embedded_voting.manipulation.voter.general import SingleVoterManipulation, SingleVoterManipulationExtension
from embedded_voting.manipulation.coalition.general import ManipulationCoalition
from embedded_voting.manipulation.coalition.ordinal import ManipulationCoalitionExtension
from embedded_voting.manipulation.voter.kapproval import SingleVoterManipulationKApp
from embedded_voting.manipulation.voter.irv import SingleVoterManipulationIRV
from embedded_voting.profile.ParametricProfile import ParametricProfile
from embedded_voting.scoring.singlewinner.ordinal import BordaExtension
from embedded_voting.scoring.singlewinner.svd import SVDSum, SVDNash
import numpy as np
from embedded_voting.profile.Profile import Profile
import matplotlib.pyplot as plt


def test_single_voter():
    my_profile = Profile(4, 3)
    my_profile.add_voters(np.random.rand(5, 3), [[1, 0.5, 0.2, 0.1]]*5)
    manipulation = SingleVoterManipulation(my_profile)
    manipulation(SVDSum())
    assert manipulation.is_manipulable_ is False

    my_profile = Profile(3, 2)
    my_profile.add_voters([[1, 0]]*3 + [[0, 1]], [[1, .9, .1]]*3 + [[.9, 1, .1]])
    manipulation = SingleVoterManipulationExtension(my_profile, BordaExtension(my_profile))
    manipulation(SVDSum())
    assert manipulation.is_manipulable_ is True

    my_profile = Profile(3, 2)
    my_profile.add_voters([[1, 0]]*3 + [[0, 1]], [[1, .9, .1]]*3 + [[.9, 1, .1]])
    manipulation = SingleVoterManipulationKApp(my_profile, 2)
    manipulation(SVDSum())
    assert manipulation.is_manipulable_ is True

    np.random.seed(45533)
    my_profile = ParametricProfile(10, 2, 10).set_parameters(.8, 0)
    manipulation = SingleVoterManipulationIRV(my_profile)
    manipulation(SVDNash())
    assert manipulation.is_manipulable_ is True


def test_coalition():
    np.random.seed(42)
    my_profile = Profile(3, 2)
    my_profile.add_voters([[1, 0]]*3 + [[0, 1]], [[1, .9, .1]]*3 + [[.9, 1, .1]])
    manipulation = ManipulationCoalition(my_profile)
    manipulation(SVDSum())

    my_profile = Profile(3, 2)
    my_profile.add_voters([[1, 0]]*3 + [[0, 1]], [[1, .9, .1]]*3 + [[.9, 1, .1]])
    manipulation = ManipulationCoalitionExtension(my_profile, BordaExtension(my_profile))
    manipulation(SVDSum())
    manipulation.trivial_manipulation(2, verbose=True)


def test_plots():
    plt.close()
    my_profile = Profile(4, 3)
    my_profile.add_voters(np.random.rand(5, 3), [[1, 0.5, 0.2, 0.1]]*5)
    manipulation = SingleVoterManipulation(my_profile, rule=SVDNash())
    manipulation.manipulation_map(map_size=5, scores_matrix=np.random.rand(3, 4))
    manipulation = ManipulationCoalition(my_profile, rule=SVDNash())
    manipulation.manipulation_map(map_size=5, scores_matrix=np.random.rand(3, 4))

