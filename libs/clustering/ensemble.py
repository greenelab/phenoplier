import os
from subprocess import call

import scipy.io as sio

import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from utils import get_temp_file_name


def reset_estimator(estimator_obj):
    for attr in dir(estimator_obj):
        if attr.startswith('_') or not attr.endswith('_'):
            continue

        delattr(estimator_obj, attr)


def anmi(ensemble, partition):
    return np.array([nmi(ensemble_member, partition) for ensemble_member in ensemble]).mean()


def aami(ensemble, partition):
    return np.array([ami(ensemble_member, partition) for ensemble_member in ensemble]).mean()


def generate_ensemble(data, clusterers: dict, attributes: list, affinity_matrix=None):
    """

    Args:
        clusterers: a dictionary with clusterers, like:
        {
            'k-means #1': KMeans(n_clusters=2),
            ...
        }
        attributes: list of attributes to save in the final dataframe

    Returns:

    """
    ensemble = []

    for clus_name, clus_obj in tqdm(clusterers.items(), total=len(clusterers)):
        # get partition
        # for agglomerative clustering both data and affinity_matrix should be given;
        # for ward linkage, data is used, and for the other linkage methods the
        # affinity_matrix is used
        if (type(clus_obj).__name__ == 'AgglomerativeClustering') and\
                (clus_obj.linkage != 'ward'):
            partition = clus_obj.fit_predict(affinity_matrix).astype(float)
        else:
            partition = clus_obj.fit_predict(data).astype(float)

        # remove from partition noisy points
        partition[partition < 0] = np.nan

        # get number of clusters
        partition_no_nan = partition[~np.isnan(partition)]
        n_clusters = np.unique(partition_no_nan).shape[0]

        # stop if n_clusters <= 1
        if n_clusters <= 1:
            reset_estimator(clus_obj)
            continue

        res = pd.Series({
            'clusterer_id': clus_name,
            'clusterer_params': str(clus_obj.get_params()),
            'partition': partition,
        })

        for attr in attributes:
            if attr == 'n_clusters' and not hasattr(clus_obj, attr):
                res[attr] = n_clusters
            else:
                res[attr] = getattr(clus_obj, attr)

        ensemble.append(res)

        reset_estimator(clus_obj)

    return pd.DataFrame(ensemble).set_index('clusterer_id')


def get_ensemble_distance_matrix(ensemble):
    # TODO: what happens if I have np.nans in the partition?

    def _compare(x, y):
        xy = np.array([x, y]).T
        xy = xy[~np.isnan(xy).any(axis=1)]
        return (xy[:, 0] != xy[:, 1]).sum() / xy.shape[0]

    return squareform(pdist(ensemble.T, _compare))
    # return pairwise_distances(ensemble.T, metric=lambda u, v: (u != v).sum() / len(u))


def eac(ensemble, k, linkage_method='average'):
    """
    Using an Evidence Accumulation method, it derives a consensus partition
    with k clusters from the clustering solutions in ensemble.

    Parameters
    ----------
    ensemble : ndarray
        Set of `p` clustering solutions generated by any clustering method. It must have
        `p` rows (number of clustering solutions) and `n` columns (number of data objects).
    k : int
        Number of clusters that will have the `x` (consensus partition).
    linkage_method : str
        Linkage criterion used for the hierachical algorithm applied on the ensemble. It supports
        any criterion provied by the `linkage` function of the `fastcluster` package.

    Returns
    -------
    x : ndarray
        Consensus clustering solution.

    References
    ----------
    [1] Fred, Ana L. N. and Jain, Anil K., "Combining Multiple Clusterings
    Using Evidence Accumulation", IEEE Trans. Pattern Anal. Mach. Intell.,
    27(6): 835-850, 2005.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> from sklearn import datasets
    >>> from sklearn.metrics import normalized_mutual_info_score as nmi
    >>> iris_data = datasets.load_iris().data
    >>> iris_data.shape
    (150, 4)
    >>> ensemble = np.array([KMeans(n_clusters=k).fit_predict(iris_data) for k in range(2,10)])
    >>> ensemble.shape
    (8, 150)
    >>> consensus_partition = eac(ensemble, 3)
    >>> iris_ref = datasets.load_iris().target
    >>> [nmi(iris_ref, p) for p in ensemble]
    [0.67932270111579207,
     0.75820572781941964,
     0.72607950712903624,
     0.7026680468096812,
     0.65459529129995175,
     0.70181110022134885,
     0.68723084369311616,
     0.66727439117973675]
    >>> nmi(iris_ref, consensus_partition)
    0.79078063459886239
    """

    y = get_ensemble_distance_matrix(ensemble)
    return AgglomerativeClustering(
        n_clusters=k,
        affinity='precomputed',
        linkage=linkage_method,
    ).fit_predict(y)
    # z = linkage(y, method=linkage_method)
    # return fcluster(z, k, criterion='maxclust')


def supraconsensus(ensemble, k, methods=None, selection_criterion=aami):
    # if methods are not provided, then use EAC methods by default (because
    # these do not depend on Octave like graph-based ones).
    if methods is None:
        eac_single_func = lambda e, k: eac(e, k, linkage_method='single')
        eac_complete_func = lambda e, k: eac(e, k, linkage_method='complete')
        eac_avg_func = lambda e, k: eac(e, k, linkage_method='average')

        methods = (eac_single_func, eac_complete_func, eac_avg_func)

    max_criterion_value = -1.0
    max_part = None

    for cmet in methods:
        part = cmet(ensemble, k)
        part_criterion_value = selection_criterion(ensemble, part)
        if part_criterion_value > max_criterion_value:
            max_criterion_value = part_criterion_value
            max_part = part

    return max_part