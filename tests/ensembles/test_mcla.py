import unittest
import numpy as np
import sklearn.datasets as datasets
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans

from clustering.ensemble import mcla


class MCLATest(unittest.TestCase):
    def test_simple(self):
        # Prepare
        ensemble = np.array(
            [
                [1, 1, 1, 2, 3, 2, 3],
                [2, 2, 2, 3, 3, 3, 1],
                [1, 1, 2, 2, 3, 3, 4],
                [2, 1, 2, 1, 3, 4, 4],
            ]
        )

        ref_partition = np.array([1, 1, 1, 2, 3, 2, 3])

        # Run
        consensus_part = mcla(ensemble, 3)

        # Validate
        assert consensus_part is not None
        assert consensus_part.shape == (7,)
        assert len(np.unique(consensus_part)) == 3
        assert ari(consensus_part, ref_partition) == 1.0

    def test_iris(self):
        # Prepare
        np.random.seed(1)

        data = datasets.load_iris().data
        ref_part = datasets.load_iris().target

        ensemble = np.array(
            [
                KMeans(n_clusters=k, init="random", n_init=1).fit_predict(data)
                for k in range(2, 10)
            ]
        )

        # Run
        consensus_part = mcla(ensemble, 3)

        # Validate
        assert consensus_part is not None
        assert consensus_part.shape == (150,)
        assert len(np.unique(consensus_part)) == 3
        assert ari(ref_part, consensus_part) >= 0.70

        # Check that the ARI obtained by the consensus partition is
        # greater than any ensemble member
        for ensemble_member in ensemble:
            assert ari(ref_part, consensus_part) > ari(ref_part, ensemble_member)

    def test_circles(self):
        # Prepare
        np.random.seed(0)

        data, ref_part = datasets.make_circles(n_samples=150, factor=0.3, noise=0.05)

        ensemble = np.array(
            [
                KMeans(n_clusters=k, init="random", n_init=1).fit_predict(data)
                for k in range(2, 15)
            ]
        )

        # Run
        consensus_part = mcla(ensemble, 2)

        # Validate
        assert consensus_part is not None
        assert consensus_part.shape == (150,)
        assert len(np.unique(consensus_part)) == 2
        ari_value = ari(ref_part, consensus_part)
        assert ari_value >= 0.95, ari_value

        # Check that the ARI obtained by the consensus partition is
        # greater than any ensemble member
        for ensemble_member in ensemble:
            assert ari(ref_part, consensus_part) > ari(ref_part, ensemble_member)
