from pathlib import Path
from IPython.display import display

import pandas as pd
from sklearn.metrics import pairwise_distances

from entity import Trait


def _predict(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir,
    prediction_function,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir:
        prediction_function:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    print(f"  predicting...", end="")

    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = Path(
        output_dir, f"{phenomexcan_input_file.stem}-prediction_scores.h5"
    ).resolve()

    if output_file.exists() and not force_run:
        print(" already run")
        return

    print("")

    drug_disease_assocs = prediction_function(lincs_projection, phenomexcan_projection)
    print(f"    shape: {drug_disease_assocs.shape}")
    drug_disease_assocs = Trait.map_to_doid(
        drug_disease_assocs, preferred_doid_list, combine="max"
    )
    print(f"    shape (after DOID map): {drug_disease_assocs.shape}")
    assert drug_disease_assocs.index.is_unique
    assert drug_disease_assocs.columns.is_unique

    # build classifier data
    print(f"  building classifier data...")
    classifier_data = (
        drug_disease_assocs.unstack()
        .reset_index()
        .rename(columns={"level_0": "trait", "perturbagen": "drug", 0: "score"})
    )
    assert classifier_data.shape == classifier_data.dropna().shape
    print(f"    shape: {classifier_data.shape}")
    display(classifier_data.describe())

    # save
    print(f"    saving to: {str(output_file)}")

    classifier_data.to_hdf(output_file, mode="w", complevel=4, key="prediction")

    pd.Series(
        {
            "method": base_method_name,
            "data": phenomexcan_input_file.stem,
        }
    ).to_hdf(output_file, mode="r+", key="metadata")


def predict_dotprod(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir_base:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    output_dir = output_dir_base / "dotprod"

    def _func(drugs_data, gene_assoc_data):
        return drugs_data.T.dot(gene_assoc_data)

    _predict(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
    )


def predict_dotprod_neg(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir_base:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    output_dir = output_dir_base / "dotprod_neg"

    def _func(drugs_data, gene_assoc_data):
        return -1.0 * drugs_data.T.dot(gene_assoc_data)

    _predict(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
    )


def _compute_pearson_distance(x, y):
    """
    TODO: complete

    Args:
        x:
        y:

    Returns:

    """
    return pd.DataFrame(
        data=pairwise_distances(x.T, y.T, metric="correlation"),
        index=x.columns.copy(),
        columns=y.columns.copy(),
    )


def predict_pearson(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir_base:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    output_dir = output_dir_base / "pearson"

    def _func(drugs_data, gene_assoc_data):
        return 1 - _compute_pearson_distance(drugs_data, gene_assoc_data)

    _predict(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
    )


def predict_pearson_neg(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir_base:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    output_dir = output_dir_base / "pearson_neg"

    def _func(drugs_data, gene_assoc_data):
        return _compute_pearson_distance(drugs_data, gene_assoc_data)

    _predict(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
    )


def predict_spearman(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir_base:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    from sklearn.metrics import pairwise_distances

    output_dir = output_dir_base / "spearman"

    def _func(drugs_data, gene_assoc_data):
        return 1 - _compute_pearson_distance(drugs_data.rank(), gene_assoc_data.rank())

    _predict(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
    )


def predict_spearman_neg(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
):
    """
    TODO: complete

    Args:
        lincs_projection:
        phenomexcan_input_file:
        phenomexcan_projection:
        output_dir_base:
        base_method_name:
        preferred_doid_list:
        force_run:

    Returns:

    """
    from sklearn.metrics import pairwise_distances

    output_dir = output_dir_base / "spearman_neg"

    def _func(drugs_data, gene_assoc_data):
        return _compute_pearson_distance(drugs_data.rank(), gene_assoc_data.rank())

    _predict(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
    )
