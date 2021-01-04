import unittest
import numpy as np
import sklearn.datasets as datasets
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans

from clustering.ensemble import eac, cspa, hgpa, mcla
from clustering.ensemble import supraconsensus
from clustering.ensemble import anmi


class SupraconsensusTest(unittest.TestCase):
    def test_iris(self):
        # Prepare
        np.random.seed(1)

        data = datasets.load_iris().data
        ref_part = datasets.load_iris().target

        ensemble = np.array([KMeans(n_clusters=k, init='random', n_init=1).fit_predict(data)
                             for k in range(2, 20)])

        # Run
        consensus_part = supraconsensus(ensemble, 3)

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
        np.random.seed(1)

        data, ref_part = datasets.make_circles(n_samples=150, factor=.3, noise=.05)

        ensemble = np.array([KMeans(n_clusters=k, init='random', n_init=1).fit_predict(data)
                             for k in range(2, 20)])

        # Run
        consensus_part = supraconsensus(ensemble, 2)

        # Validate
        assert consensus_part is not None
        assert consensus_part.shape == (150,)
        assert len(np.unique(consensus_part)) == 2
        assert ari(ref_part, consensus_part) >= 0.95

        # Check that the ARI obtained by the consensus partition is
        # greater than any ensemble member
        for ensemble_member in ensemble:
            assert ari(ref_part, consensus_part) > ari(ref_part, ensemble_member)

    def test_anmi_max_eac(self):
        # Prepare
        np.random.seed(1)

        data, ref_part = datasets.make_circles(n_samples=150, factor=.3, noise=.05)

        ensemble = np.array([KMeans(n_clusters=k, init='random', n_init=1).fit_predict(data)
                             for k in range(2, 10)])

        # Run
        consensus_part = supraconsensus(ensemble, 2)

        eac_single_part = eac(ensemble, 2, linkage_method='single')
        eac_complete_part = eac(ensemble, 2, linkage_method='complete')
        eac_average_part = eac(ensemble, 2, linkage_method='average')

        # supraconsensus partition should be one of the others
        assert ari(consensus_part, eac_single_part) == 1.0 or \
               ari(consensus_part, eac_complete_part) == 1.0 or \
               ari(consensus_part, eac_average_part) == 1.0

        # ANMI should be greater or equal for the supraconsensus
        assert anmi(ensemble, consensus_part) >= anmi(ensemble, eac_single_part)
        assert anmi(ensemble, consensus_part) >= anmi(ensemble, eac_complete_part)
        assert anmi(ensemble, consensus_part) >= anmi(ensemble, eac_average_part)

    def test_anmi_max_graphs(self):
        # Prepare
        np.random.seed(1)

        data, ref_part = datasets.make_circles(n_samples=150, factor=.3, noise=.05)

        ensemble = np.array([KMeans(n_clusters=k, init='random', n_init=1).fit_predict(data)
                             for k in range(2, 6)])

        # Run
        consensus_part = supraconsensus(ensemble, 2, methods=(cspa, hgpa, mcla))

        cspa_part = cspa(ensemble, 2)
        hgpa_part = hgpa(ensemble, 2)
        mcla_part = mcla(ensemble, 2)

        # supraconsensus partition should be one of the others
        assert ari(consensus_part, cspa_part) == 1.0 or \
               ari(consensus_part, hgpa_part) == 1.0 or \
               ari(consensus_part, mcla_part) == 1.0

        # ANMI should be greater or equal for the supraconsensus
        assert anmi(ensemble, consensus_part) >= anmi(ensemble, cspa_part)
        assert anmi(ensemble, consensus_part) >= anmi(ensemble, hgpa_part)
        assert anmi(ensemble, consensus_part) >= anmi(ensemble, mcla_part)