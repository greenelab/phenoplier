import warnings

import numpy as np

from sklearn.cluster import SpectralClustering


class DeltaSpectralClustering(SpectralClustering):
    """
    TODO: complete
    """

    def __init__(self, delta=1.0, **kwargs):
        """
        TODO: complete

        - affinity is replaced by "precomputed"

        Args:
            delta:
            **kwargs:
        """

        if "affinity" in kwargs:
            warnings.warn(
                "The 'affinity' parameter was specified; it will be overridden with 'precomputed'"
            )

        kwargs["affinity"] = "precomputed"

        super().__init__(**kwargs)

        self.delta = delta

    def fit(self, X, y=None):
        """
        TODO: complete

        Args:
            X: it has to be a distance matrix!
            y: not used

        Returns:

        """
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                f"The data has to be a squared distance matrix;"
                + " current shape is {X.shape}"
            )

        # This Gaussian kernel is suggested in the sklearn documentation for
        # SpectralClustering. It converts a distance matrix to a similarity
        # matrix.
        X = np.exp(-(X ** 2) / (2.0 * self.delta ** 2))
        return super().fit(X, y)
