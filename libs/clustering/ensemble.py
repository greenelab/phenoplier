import pandas as pd
from tqdm import tqdm


def generate_ensemble(data, clusterers: dict, attributes: list):
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
        res = pd.Series({
            'clusterer_id': clus_name,
            'clusterer_params': str(clus_obj.get_params()),
            'partition': clus_obj.fit_predict(data)
        })

        for attr in attributes:
            res[attr] = getattr(clus_obj, attr)

        ensemble.append(res)

    return pd.DataFrame(ensemble).set_index('clusterer_id')
