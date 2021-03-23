from embedded_voting.profile.Profile import Profile
import numpy as np


class AutoProfile(Profile):

    def add_voters_auto(self, samples, scores, n_dim=0, normalize_score=True):
        samples_total = np.concatenate([samples, scores], axis=1)
        cov = np.cov(samples_total)

        if n_dim == 0:
            eigen_values, _ = np.linalg.eig(cov)
            eigen_values_participation = eigen_values / np.sum(eigen_values)
            while n_dim < len(eigen_values_participation) and eigen_values_participation[n_dim] > 0.01:
                n_dim += 1

        cov_transpose = np.cov(samples_total.T)
        _, eigen_vectors = np.linalg.eig(cov_transpose)
        projection_matrix = (eigen_vectors.T[:][:n_dim]).T
        embs = samples_total.dot(projection_matrix)

        self.n_dim = n_dim
        self.embeddings = np.zeros((0, n_dim))

        if normalize_score:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        embs = np.real(embs)
        self.add_voters(embs, scores)
        return self

    def add_voters_cov(self, train_samples, scores):
        samples_total = np.concatenate([train_samples, scores], axis=1)
        cov = np.cov(samples_total)
        self.add_voters(cov, scores, normalize_embs=False)
        return self
