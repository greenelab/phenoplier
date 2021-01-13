import numpy as np

from sklearn.cluster import SpectralClustering


class DeltaSpectralClustering(SpectralClustering):
    """
    TODO: complete
    """

    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def fit(self, X, y=None):
        # This Gaussian kernel is suggested in the sklearn documentation for
        # SpectralClustering
        X = np.exp(-(X ** 2) / (2.0 * self.delta ** 2))
        return super().fit(X, y)
