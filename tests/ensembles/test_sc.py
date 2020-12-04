import unittest
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
import numpy as np

from clustering.ensemble import sc_consensus, get_ensemble_distance_matrix

iris_data = datasets.load_iris().data
iris_ref = datasets.load_iris().target


def test_iris_ensemble_with_2to10():
    # Prepare
    np.random.seed(1)

    ensemble = np.array(
        [
            KMeans(n_clusters=k, init="random", n_init=1).fit_predict(iris_data)
            for k in range(2, 10)
        ]
    )

    # Run
    ensemble_distance_matrix = get_ensemble_distance_matrix(ensemble)
    consensus_partition = sc_consensus(ensemble_distance_matrix, 3)

    # Validate
    assert len(np.unique(consensus_partition)) == 3
    assert consensus_partition is not None
    assert 150 == len(consensus_partition)

    # Check that the ARI obtained by the consensus partition is
    # greater than any ensemble member
    for ensemble_member in ensemble:
        assert ari(iris_ref, consensus_partition) > ari(iris_ref, ensemble_member)
