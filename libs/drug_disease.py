from pathlib import Path
from IPython.display import display

import pandas as pd
from sklearn.metrics import pairwise_distances

from entity import Trait


def _zero_nontop_genes(trait_vector, n_top, use_abs=False):
    if use_abs:
        df = trait_vector.abs()
    else:
        df = trait_vector

    trait_top_genes = df.sort_values(ascending=False).head(n_top).index

    trait_vector = trait_vector.copy()
    trait_vector[~trait_vector.index.isin(trait_top_genes)] = 0.0

    return trait_vector


def _predict(
    lincs_projection,
    phenomexcan_input_file,
    phenomexcan_projection,
    output_dir,
    prediction_function,
    base_method_name,
    preferred_doid_list,
    force_run,
    n_top_conditions=None,
    use_abs=False,
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
    output_file_suffix = "all_genes"
    if n_top_conditions is not None:
        output_file_suffix = f"top_{n_top_conditions}_genes"

    print(f"  predicting {output_file_suffix}...", end="")

    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = Path(
        output_dir,
        f"{phenomexcan_input_file.stem}-{output_file_suffix}-prediction_scores.h5",
    ).resolve()

    if output_file.exists() and not force_run:
        print(" already run")
        return

    print("")

    if n_top_conditions is not None:
        # FIXME: this only works with the current dotprod/dotprod_neg methods
        phenomexcan_projection = phenomexcan_projection.apply(
            lambda x: _zero_nontop_genes(x, n_top_conditions, use_abs)
        )

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

    # FIXME: I think another way to put this is a list with the dict, instead of
    # a list for each value as now
    pd.DataFrame(
        {
            "method": [base_method_name],
            "n_top_genes": [-1.0 if n_top_conditions is None else n_top_conditions],
            "data": [phenomexcan_input_file.stem],
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
    n_top_conditions=None,
    use_abs=False,
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
        n_top_conditions,
        use_abs,
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
