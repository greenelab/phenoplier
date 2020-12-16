import pandas as pd
from IPython.display import display


def get_data_proj(dim_red_method_obj, data: pd.DataFrame) -> pd.DataFrame:
    """
    It applies a dimensionality reduction method on a data set, preserving row
    and column names.

    Args:
        dim_red_method_obj: any object that has a fit_transform method (using
            sklearn API) to perform dimensionality reduction. The __name__
            attribute of this object is used to rename the columns of the final
            data frame.
        data: the data set where the dimensionality method will be applied to.

    Returns:
        a new data frame with the data projected using the method provided.
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
    Performs PCA dimensionality reduction with the given options.

    Args:
        data: data where PCA will be applied to.
        options: PCA options from sklearn's PCA implementation.

    Returns:
        a new data frame with the PCA version of the data.
    """
    from sklearn.decomposition import PCA

    pca_obj = PCA(**options)
    return get_data_proj(pca_obj, data)


def get_umap_proj(data: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    Returns a UMAP representation of data with the options specified in options.

    Args:
        data: data where UMAP will be applied to.
        options: UMAP options from umap-learn implementation.

    Returns:
        a new data frame with the UMAP version of the data.
    """
    from umap import UMAP

    umap_obj = UMAP(**options)
    return get_data_proj(umap_obj, data)
