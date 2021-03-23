"""Top-level package for Embedded Voting."""

__author__ = """Théo Delemazure"""
__email__ = 'theo.delemazure@ens.fr'
__version__ = '0.1.0'


from embedded_voting.profile.Profile import *
from embedded_voting.profile.ParametricProfile import *
from embedded_voting.profile.MovingVoter import *
from embedded_voting.scoring import *
from embedded_voting.manipulation import *
from embedded_voting.utils.plots import *
from embedded_voting.algorithm_aggregation.score_generator import *
from embedded_voting.algorithm_aggregation.maximumlikelihood import *
from embedded_voting.algorithm_aggregation.auto_embeddings import *
