import warnings

import numpy as np

from sklearn.cluster import SpectralClustering


class DeltaSpectralClustering(SpectralClustering):
    """
    It extends SpectralClustering by accepting a distance matrix as input and
    applying a Gaussian kernel before fitting.
    """

    def __init__(self, delta=1.0, **kwargs):
        """
        The same arguments of :obj:`sklearn.cluster.SpectralClustering`, plus
        a ``delta`` parameter. If ``affinity`` is provided, it will be overridden
        to "precomputed".

        Args:
            delta: it represents the width of the Gaussian kernel applied to the
                input distance matrix before fitting.
            **kwargs: any parameter accepted by
                :obj:`sklearn.cluster.SpectralClustering`. If ``affinity`` is
                provided, a warning is raised and it is overridden by
                "precomputed".
        """

        if "affinity" in kwargs:
            warnings.warn(
                "The 'affinity' parameter was specified; it will be overridden "
                + "with 'precomputed'"
            )

        kwargs["affinity"] = "precomputed"

        super().__init__(**kwargs)

        self.delta = delta

    def fit(self, X, y=None):
        """
        Performs spectral clustering from an affinity matrix derived from the
        distance matrix provided. It applies a Gaussian kernel before fitting::

            np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

        where the ``delta`` represents the width of the Gaussian kernel.

        Args:
            X: array-like, shape (n_samples, n_samples)
                Distance matrix for all instances to cluster (0 means identical
                instances, and high values mean very dissimilar instances).
            y: Ignored
                Not used, present here for API consistency by convention.

        Returns:
            self
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                "The data has to be a squared distance matrix;"
                + f" current shape is {X.shape}"
            )

        # This Gaussian kernel is suggested in the sklearn documentation for
        # SpectralClustering. It converts a distance matrix to a similarity
        # matrix.
        X = np.exp(-(X ** 2) / (2.0 * self.delta ** 2))

        return super().fit(X, y)
