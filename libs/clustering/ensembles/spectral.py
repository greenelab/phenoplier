from clustering.ensembles.utils import get_ensemble_distance_matrix
from clustering.methods import DeltaSpectralClustering


def scc(
    ensemble,
    k: int,
    delta: float = 1.0,
    ensemble_is_coassoc_matrix: bool = False,
    **kwargs,
):
    """
    TODO: complete

    Args:
        ensemble:
        k:
        delta:
        ensemble_is_coassoc_matrix:
        **kwargs:

    Returns:

    """

    if ensemble_is_coassoc_matrix:
        data = ensemble
    else:
        data = get_ensemble_distance_matrix(ensemble)

    return DeltaSpectralClustering(
        n_clusters=k,
        delta=delta,
        **kwargs,
    ).fit_predict(data)
