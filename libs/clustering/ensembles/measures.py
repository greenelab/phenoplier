import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_mutual_info_score as ami,
    adjusted_rand_score as ari,
)

from clustering.utils import compare_arrays


def anmi(ensemble, partition):
    """
    Computes the average of the normalized mutual information (NMI) between each
    ensemble member and a given partition.

    Args:
        ensemble:
            A numpy array representing a set of clustering solutions on the same
            data. Each row is a clustering solution (partition) and columns are
            objects.
        partition:
            A 1D numpy array with a consensus partition.

    Returns:
        A numerical value with the average of the NMI between each ensemble member
        and the consensus partition.
    """
    return np.array(
        [
            compare_arrays(ensemble_member, partition, nmi)
            for ensemble_member in ensemble
        ]
    ).mean()


def aami(ensemble, partition):
    """
    Computes the average of the adjusted mutual information (AMI) between each
    ensemble member and a given partition.

    Args:
        ensemble:
            A numpy array representing a set of clustering solutions on the same
            data. Each row is a clustering solution (partition) and columns are
            objects.
        partition:
            A 1D numpy array with a consensus partition.

    Returns:
        A numerical value with the average of the AMI between each ensemble
        member and the consensus partition.
    """
    return np.array(
        [
            compare_arrays(ensemble_member, partition, ami)
            for ensemble_member in ensemble
        ]
    ).mean()


def aari(ensemble, partition):
    """
    Computes the average of the adjusted rand index (ARI) between each ensemble
    member and a given partition.

    Args:
        ensemble:
            A numpy array representing a set of clustering solutions on the same
            data. Each row is a clustering solution (partition) and columns are
            objects.
        partition:
            A 1D numpy array with a consensus partition.

    Returns:
        A numerical value with the average of the ARI between each ensemble
        member and the consensus partition.
    """
    return np.array(
        [
            compare_arrays(ensemble_member, partition, ari)
            for ensemble_member in ensemble
        ]
    ).mean()
