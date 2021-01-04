import unittest
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
import numpy as np

from clustering.ensemble import eac, get_ensemble_distance_matrix

iris_data = datasets.load_iris().data
iris_ref = datasets.load_iris().target


class CoassocEACTest(unittest.TestCase):
    def test_get_ensemble_distance_matrix(self):
        # Prepare
        ensemble = np.array(
            [
                [1, 1, 1, 2],
                [2, 2, 2, 3],
                [1, 1, 2, 2],
                [3, 1, 2, 2],
                [1, 1, 3, 2],
                [2, 1, 2, 1],
            ]
        )

        ensemble_distance_matrix = get_ensemble_distance_matrix(ensemble)
        assert ensemble_distance_matrix is not None
        assert ensemble_distance_matrix.shape == (4, 4)

        np.testing.assert_array_equal(
            np.around(ensemble_distance_matrix, 2),
            np.around(
                np.array(
                    [
                        [
                            0,
                            2 / 6.0,
                            3 / 6.0,
                            6 / 6.0,
                        ],
                        [
                            2 / 6.0,
                            0,
                            4 / 6.0,
                            5 / 6.0,
                        ],
                        [3 / 6.0, 4 / 6.0, 0, 4 / 6.0],
                        [
                            6 / 6.0,
                            5 / 6.0,
                            4 / 6.0,
                            0,
                        ],
                    ]
                ),
                2,
            ),
        )

    def test_get_ensemble_distance_matrix_with_nan(self):
        # Prepare
        ensemble = np.array(
            [
                [1, 1, 1, 2],
                [2, 2, 2, 3],
                [np.nan, 1, 2, 2],
                [3, 1, 2, 2],
                [1, 1, 3, 2],
                [2, 1, 2, 1],
            ]
        )

        ensemble_distance_matrix = get_ensemble_distance_matrix(ensemble)
        assert ensemble_distance_matrix is not None
        assert ensemble_distance_matrix.shape == (4, 4)

        np.testing.assert_array_equal(
            np.around(ensemble_distance_matrix, 2),
            np.around(
                np.array(
                    [
                        [
                            0,
                            2 / 5.0,
                            2 / 5.0,
                            5 / 5.0,
                        ],
                        [
                            2 / 5.0,
                            0,
                            4 / 6.0,
                            5 / 6.0,
                        ],
                        [2 / 5.0, 4 / 6.0, 0, 4 / 6.0],
                        [
                            5 / 5.0,
                            5 / 6.0,
                            4 / 6.0,
                            0,
                        ],
                    ]
                ),
                2,
            ),
        )

    def test_get_ensemble_distance_matrix_with_more_nan(self):
        # Prepare
        ensemble = np.array(
            [
                [1, 1, 1, 2],
                [2, 2, 2, 3],
                [np.nan, 1, 2, 2],
                [3, np.nan, 2, 2],
                [1, 1, 3, 2],
                [2, 1, 2, 1],
            ]
        )

        ensemble_distance_matrix = get_ensemble_distance_matrix(ensemble)
        assert ensemble_distance_matrix is not None
        assert ensemble_distance_matrix.shape == (4, 4)

        np.testing.assert_array_equal(
            np.around(ensemble_distance_matrix, 2),
            np.around(
                np.array(
                    [
                        [
                            0,
                            1 / 4.0,
                            2 / 5.0,
                            5 / 5.0,
                        ],
                        [
                            1 / 4.0,
                            0,
                            3 / 5.0,
                            4 / 5.0,
                        ],
                        [2 / 5.0, 3 / 5.0, 0, 4 / 6.0],
                        [
                            5 / 5.0,
                            4 / 5.0,
                            4 / 6.0,
                            0,
                        ],
                    ]
                ),
                2,
            ),
        )

    def test_get_ensemble_distance_matrix_all_nan(self):
        # Prepare
        ensemble = np.array(
            [
                [1, np.nan, 1, 2],
                [np.nan, 2, 2, 3],
                [np.nan, 1, 2, 2],
                [3, np.nan, 2, 2],
                [np.nan, 1, 3, 2],
                [np.nan, 1, 2, 1],
            ]
        )

        ensemble_distance_matrix = get_ensemble_distance_matrix(ensemble)
        assert ensemble_distance_matrix is not None
        assert ensemble_distance_matrix.shape == (4, 4)

        np.testing.assert_array_equal(
            np.around(ensemble_distance_matrix, 2),
            np.around(
                np.array(
                    [
                        [
                            0,
                            np.nan,
                            1 / 2.0,
                            2 / 2.0,
                        ],
                        [
                            np.nan,
                            0,
                            3 / 4.0,
                            3 / 4.0,
                        ],
                        [1 / 2.0, 3 / 4.0, 0, 4 / 6.0],
                        [
                            2 / 2.0,
                            3 / 4.0,
                            4 / 6.0,
                            0,
                        ],
                    ]
                ),
                2,
            ),
        )

    def test_iris_ensemble_with_2to10(self):
        # Prepare
        np.random.seed(1)

        ensemble = np.array(
            [
                KMeans(n_clusters=k, init="random", n_init=1).fit_predict(iris_data)
                for k in range(2, 10)
            ]
        )

        # Run
        consensus_partition = eac(ensemble, 3)

        # Validate
        assert len(np.unique(consensus_partition)) == 3
        self.assertIsNotNone(consensus_partition)
        self.assertEqual(150, len(consensus_partition))

        # Check that the ARI obtained by the consensus partition is
        # greater than any ensemble member
        for ensemble_member in ensemble:
            self.assertGreaterEqual(
                ari(iris_ref, consensus_partition), ari(iris_ref, ensemble_member)
            )
