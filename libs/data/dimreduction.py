import pandas as pd
from IPython.display import display


def get_data_proj(dim_red_method_obj, data: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: complete

    Args:
        dim_red_method_obj:
        data:

    Returns:

    """
    method_name = type(dim_red_method_obj).__name__
    display(f'{method_name} object: {dim_red_method_obj}')

    proj_data = dim_red_method_obj.fit_transform(data)

    return pd.DataFrame(
        data=proj_data,
        index=data.index.copy(),
        columns=[f'{method_name}{i+1}' for i in range(proj_data.shape[1])]
    )


def get_pca_proj(data: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    TODO: complete

    Args:
        data:
        options:

    Returns:

    """
    from sklearn.decomposition import PCA

    pca_obj = PCA(**options)
    return get_data_proj(pca_obj, data)


def get_umap_proj(data: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    Returns a UMAP representation of orig_data with the options specified
    globally in DR_OPTIONS.

    TDOO: complete

    Args:
        data:
        options:

    Returns:

    """
    from umap import UMAP

    umap_obj = UMAP(**options)
    return get_data_proj(umap_obj, data)
