import numpy as np
from IPython.display import display
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def compute_performance(data, labels, data_distance_matrix=None):
    """
    TODO: complete

    Args:
        data:
        labels:
        data_distance_matrix:

    Returns:

    """

    # From sklearn website: The best value is 1 and the worst value is -1.
    # Values near 0 indicate overlapping clusters. Negative values generally
    # indicate that a sample has been assigned to the wrong cluster, as a
    # different cluster is more similar.
    if data_distance_matrix is not None:
        si_score = silhouette_score(data_distance_matrix, labels, metric="precomputed")
    else:
        si_score = silhouette_score(data, labels)
    display(f"Silhouette (higher is better): {si_score:.3f}")

    # From sklearn website: It is also known as the Variance Ratio Criterion.
    # The score is defined as ratio between the within-cluster dispersion and
    # the between-cluster dispersion.
    ch_score = calinski_harabasz_score(data, labels)
    display(f"Calinski-Harabasz (higher is better): {ch_score:.3f}")

    # From sklearn website: The score is defined as the average similarity
    # measure of each cluster with its most similar cluster, where similarity is
    # the ratio of within-cluster distances to between-cluster distances. Thus,
    # clusters which are farther apart and less dispersed will result in a
    # better score. The minimum score is zero, with lower values indicating
    # better clustering.
    db_score = davies_bouldin_score(data, labels)
    display(f"Davies-Bouldin (lower is better): {db_score:.3f}")


def reset_estimator(estimator_obj):
    """
    Resets an sklearn estimator by removing all attributes generated during
    fitting.

    Args:
        estimator_obj: an sklearn estimator

    Returns:
        It doesn't return anything (always None).
    """
    for attr in dir(estimator_obj):
        if attr.startswith("_") or not attr.endswith("_"):
            continue

        delattr(estimator_obj, attr)


def compare_arrays(x, y, comp_func):
    """
    It compares non-nan values from two numpy arrays using a specified
    function that returns a numerical value.

    In this module, these two arrays are always data partitions (generated
    from a clustering algorithm, and specifying the cluster each object
    belongs to), and the return value is a similarity measure between them.

    Args:
        x: a 1D numpy array.
        y: a 1D numpy array.
        comp_func: any function that accepts two arguments (numpy arrays) and
            returns a numerical value.

    Returns:
        Any numerical value representing, for instance, the similarity between
        the two arrays.
    """
    xy = np.array([x, y]).T
    xy = xy[~np.isnan(xy).any(axis=1)]
    return comp_func(xy[:, 0], xy[:, 1])
