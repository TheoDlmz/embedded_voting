import numpy as np
from embedded_voting.embeddings.embeddings import Embeddings
from embedded_voting.embeddings.embeddings_generator import EmbeddingsGenerator


class EmbeddingsGeneratorRandom(EmbeddingsGenerator):
    """
    Create random embeddings

    Examples
    --------
    >>> np.random.seed(42)
    >>> generator = EmbeddingsGeneratorRandom(10, 2)
    >>> generator()
    Embeddings([[0.96337365, 0.26816265],
                [0.39134578, 0.92024371],
                [0.70713157, 0.70708199],
                [0.89942118, 0.43708299],
                [0.65433791, 0.75620229],
                [0.70534506, 0.70886413],
                [0.1254653 , 0.99209801],
                [0.95076   , 0.30992809],
                [0.95508537, 0.29633078],
                [0.54080587, 0.84114744]])
    """
    def __call__(self, *args):
        embs = np.abs(np.random.randn(self.n_voters, self.n_dim))
        return Embeddings(embs, norm=True)
