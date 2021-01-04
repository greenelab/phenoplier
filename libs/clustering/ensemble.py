"""
Contains functions to generate and combine a clustering ensemble.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from tqdm import tqdm


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


def generate_ensemble(data, clusterers: dict, attributes: list, affinity_matrix=None):
    """
    It generates an ensemble from the data given a set of clusterers (a
    clusterer is an instance of a clustering algorithm with a fixed set of
    parameters).

    Args:
        data:
            A numpy array, pandas dataframe, or any other structure supported
            by the clusterers as data input.
        clusterers:
            A dictionary with clusterers specified in this format: { 'k-means
            #1': KMeans(n_clusters=2), ... }
        attributes:
            A list of attributes to save in the final dataframe; for example,
            including "n_clusters" will extract this attribute from the
            estimator and include it in the final dataframe returned.
        affinity_matrix:
            If the clustering algorithm is AgglomerativeClustering (from
            sklearn) and the linkage method is different than ward (which only
            support euclidean distance), the affinity_matrix is given as data
            input to the estimator instead of data.

    Returns:
        A pandas DataFrame with all the partitions generated by the clusterers.
        Columns include the clusterer name/id, the partition, the estimator
        parameters (obtained with the get_params() method) and any other
        attribute specified.
    """
    ensemble = []

    for clus_name, clus_obj in tqdm(clusterers.items(), total=len(clusterers)):
        # get partition
        #
        # for agglomerative clustering both data and affinity_matrix should be
        # given; for ward linkage, data is used, and for the other linkage
        # methods the affinity_matrix is used
        if (type(clus_obj).__name__ == "AgglomerativeClustering") and (
            clus_obj.linkage != "ward"
        ):
            partition = clus_obj.fit_predict(affinity_matrix).astype(float)
        else:
            partition = clus_obj.fit_predict(data).astype(float)

        # remove from partition noisy points (for example, if using DBSCAN)
        partition[partition < 0] = np.nan

        # get number of clusters
        partition_no_nan = partition[~np.isnan(partition)]
        n_clusters = np.unique(partition_no_nan).shape[0]

        # stop if n_clusters <= 1
        if n_clusters <= 1:
            reset_estimator(clus_obj)
            continue

        res = pd.Series(
            {
                "clusterer_id": clus_name,
                "clusterer_params": str(clus_obj.get_params()),
                "partition": partition,
            }
        )

        for attr in attributes:
            if attr == "n_clusters" and not hasattr(clus_obj, attr):
                res[attr] = n_clusters
            else:
                res[attr] = getattr(clus_obj, attr)

        ensemble.append(res)

        # for some estimators such as DBSCAN this is needed, because otherwise
        # the estimator saves references of huge data structures not needed in
        # this context
        reset_estimator(clus_obj)

    return pd.DataFrame(ensemble).set_index("clusterer_id")


def get_ensemble_distance_matrix(ensemble, n_jobs=1):
    """
    Given an ensemble, it computes the coassociation matrix (a distance matrix
    for all objects using the ensemble information). For each object pair, the
    coassociation matrix contains the percentage of times the pair of objects
    was clustered together in the ensemble.

    Args:
        ensemble:
            A numpy array representing a set of clustering solutions on the same
            data. Each row is a clustering solution (partition) and columns are
            objects.
        n_jobs:
            The number of jobs used by the pairwise_distance matrix from
            sklearn.

    Returns:
        A numpy array representing a square distance matrix for all objects
        (coassociation matrix).
    """

    def _compare(x, y):
        xy = np.array([x, y]).T
        xy = xy[~np.isnan(xy).any(axis=1)]
        return (xy[:, 0] != xy[:, 1]).sum() / xy.shape[0]

    return pairwise_distances(
        ensemble.T, metric=_compare, n_jobs=n_jobs, force_all_finite="allow-nan"
    )


def eac(
    ensemble,
    k: int,
    linkage_method: str = "average",
    ensemble_is_coassoc_matrix: bool = False,
):
    """
    Using an Evidence Accumulation method (EAC) [1], it derives a consensus
    partition with k clusters from the clustering solutions in ensemble.

    [1] Fred, Ana L. N. and Jain, Anil K., "Combining Multiple Clusterings Using
    Evidence Accumulation", IEEE Trans. Pattern Anal. Mach. Intell., 27(6):
    835-850, 2005.

    Args:
        ensemble:
            Set of `p` clustering solutions generated by any clustering method.
            It must have `p` rows (number of clustering solutions) and `n`
            columns (number of data objects).
        k:
            Number of clusters that will have the `x` (consensus partition).
        linkage_method:
            Linkage criterion used for the hierachical algorithm applied on the
            ensemble. It supports any criterion provied by the `linkage`
            function of the `fastcluster` package.

    Returns:
        A 1D numpy array with the consensus clustering solution.
    """

    if ensemble_is_coassoc_matrix:
        y = ensemble
    else:
        y = get_ensemble_distance_matrix(ensemble)

    return AgglomerativeClustering(
        n_clusters=k,
        affinity="precomputed",
        linkage=linkage_method,
    ).fit_predict(y)


def eac_single(ensemble, k):
    """
    Shortcut to run EAC using the single linkage method on the ensemble.
    """
    return eac(ensemble, k, linkage_method="single")


def eac_single_coassoc_matrix(coassoc_matrix, k):
    """
    Shortcut to run EAC using the single linkage method on the coassociation
    matrix.
    """
    return eac(
        coassoc_matrix, k, ensemble_is_coassoc_matrix=True, linkage_method="single"
    )


def eac_complete(ensemble, k):
    """
    Shortcut to run EAC using the complete linkage method on the ensemble.
    """
    return eac(ensemble, k, linkage_method="complete")


def eac_complete_coassoc_matrix(coassoc_matrix, k):
    """
    Shortcut to run EAC using the complete linkage method on the coassociation
    matrix.
    """
    return eac(
        coassoc_matrix, k, ensemble_is_coassoc_matrix=True, linkage_method="complete"
    )


def eac_average(ensemble, k):
    """
    Shortcut to run EAC using the average linkage method on the ensemble.
    """
    return eac(ensemble, k, linkage_method="average")


def eac_average_coassoc_matrix(coassoc_matrix, k):
    """
    Shortcut to run EAC using the average linkage method on the coassociation
    matrix.
    """
    return eac(
        coassoc_matrix, k, ensemble_is_coassoc_matrix=True, linkage_method="average"
    )


def supraconsensus(
    ensemble, k, methods=None, selection_criterion=aari, n_jobs=1, use_tqdm=False
):
    """
    It combines a clustering ensemble using a set of methods that the user can
    specify. Each of these methods combines the ensemble and returns a single
    partition. This function returns the combined partition that maximizes the
    selection criterion.

    Args:
        ensemble:
            a clustering ensemble (rows are partitions, columns are objects).
        k:
            the final number of clusters for the combined partition.
        methods:
            a list of methods to apply on the ensemble; each returns a combined
            partition.
        selection_criterion:
            a function that represents the selection criterion; this function
            has to accept an ensemble as the first argument, and a partition as
            the second one.
        n_jobs:
            number of jobs.
        use_tqdm:
            ensembles/disables the use of tqdm to show a progress bar.

    Returns:
        Returns a tuple: (partition, best method name, best criterion value)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # if methods are not provided, then use EAC methods by default
    if methods is None:
        methods = (eac_single, eac_complete, eac_average)

    methods_results = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        tasks = {executor.submit(m, ensemble, k): m.__name__ for m in methods}

        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            disable=(not use_tqdm),
            ncols=100,
        ):
            method_name = tasks[future]
            part = future.result()
            criterion_value = selection_criterion(ensemble, part)

            methods_results[method_name] = {
                "partition": part,
                "criterion_value": criterion_value,
            }

    # select the best performing method according to the selection criterion
    best_method = max(
        methods_results, key=lambda x: methods_results[x]["criterion_value"]
    )
    best_method_results = methods_results[best_method]

    return (
        best_method_results["partition"],
        best_method,
        best_method_results["criterion_value"],
    )


def run_method_and_compute_agreement(method_func, ensemble_data, ensemble, k):
    """
    Runs a consensus clustering method on the ensemble data, obtains the
    consolidated partition with the desired number of clusters, and computes
    a series of performance measures.

    Args:
        method_func:
            A consensus function (first argument is either the ensemble or
            the coassociation matrix derived from the ensemble).
        ensemble_data:
            A numpy array with the ensemble data that will be given to the
            specified method. For evidence accumulation methods, this is the
            coassociation matrix (a square matrix with the distance between
            object pairs derived from the ensemble).
        ensemble:
            A numpy array representing the ensemble (partitions in rows, objects
            in columns).
        k:
            The number of clusters to obtain from the ensemble data using the
            specified method.

    Returns:
        It returns a tuple with the data partition derived from the ensemble
        data using the specified method, and some performance measures of this
        partition.
    """
    part = method_func(ensemble_data, k)

    nmi_values = np.array(
        [compare_arrays(ensemble_member, part, nmi) for ensemble_member in ensemble]
    )

    ami_values = np.array(
        [compare_arrays(ensemble_member, part, ami) for ensemble_member in ensemble]
    )

    ari_values = np.array(
        [compare_arrays(ensemble_member, part, ari) for ensemble_member in ensemble]
    )

    performance_values = {
        "ari_mean": np.mean(ari_values),
        "ari_median": np.median(ari_values),
        "ari_std": np.std(ari_values),
        "ami_mean": np.mean(ami_values),
        "ami_median": np.median(ami_values),
        "ami_std": np.std(ami_values),
        "nmi_mean": np.mean(nmi_values),
        "nmi_median": np.median(nmi_values),
        "nmi_std": np.std(nmi_values),
    }

    return part, performance_values
