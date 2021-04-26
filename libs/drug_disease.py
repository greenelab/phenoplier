"""
Contains functions to run drug-disease predictions given gene-disease
associations and drug-induced gene expression.
"""

from pathlib import Path
from IPython.display import display

import pandas as pd
from sklearn.metrics import pairwise_distances

from entity import Trait


def _zero_nontop_genes(trait_vector, n_top, use_abs=True):
    """
    It takes a pandas Series (with genes/LVs in index, and association values),
    sorts it, and leave only the top values, putting zeros in the rest.

    Args:
        trait_vector:
            A pandas Series with numerical data.
        n_top:
            Number of top genes/LVs to keep. The rest will be zeroed.
        use_abs:
            If True, then it uses the absolute value.

    Returns:
        A new pandas Series keeping only the top n_top values (the rest is
        zeroed).
    """
    if use_abs:
        df = trait_vector.abs()
    else:
        df = trait_vector

    trait_top_genes = df.sort_values(ascending=False).head(n_top).index

    trait_vector = trait_vector.copy()
    trait_vector[~trait_vector.index.isin(trait_top_genes)] = 0.0

    return trait_vector


def _predict(
    drug_gene_data,
    gene_trait_data_filename,
    gene_trait_data,
    output_dir,
    prediction_function,
    base_method_name,
    preferred_doid_list,
    force_run,
    n_top_conditions=None,
    use_abs=False,
):
    """
    Given gene-disease associations (standardized effect sizes) and the
    drug-induced gene expression (z-scores), it computes a drug-disease score
    using an approach based on a published framework for drug-repositioning
    (https://doi.org/10.1038/nn.4618).

    This function allows to specify here the prediction_function, which could
    potentially be the Pearson or Spearman correlation, but here we use the dot
    product only (see predict_dotprod_neg).

    The function saves an HDF5 file with three keys:
        * `full_prediction`: it has predictions for all drugs and traits given
            as argument.
        * `prediction`: it has predictions for all drugs, but includes traits that
            can map to Disease Ontology ID only. This is used for comparison between
            methods using the gold standard from PharmacotherapyDB.
        * `metadata`: a pandas dataframe with metadata about the run.

    Args:
        drug_gene_data:
            A pandas DataFrame containing the drug-gene associations. Genes
            must be in rows, and drugs in columns.
        gene_trait_data_filename:
            The file name where the `gene_trait_data` was read from. It is used
            to generate the output file name and save in the metadata.
        gene_trait_data:
            A pandas DataFrame containing the gene-trait associations. Genes
            must be in rows, and traits in columns.
        output_dir:
            A Path object pointing to the output directory where predictions
            will be saved.
        prediction_function:
            A prediction function that receives two arguments: the drug_gene_data
            and the gene_trait_data, and returns a pandas DataFrame with the drug-disease
            predictions (drugs in rows, traits in columns).
        base_method_name:
            A string with the name of the method being used for prediction. Currently,
            it is either "Gene-based" or "Module-based".
        preferred_doid_list:
            List of Disease Ontology IDs (DOID) that are "preferred". This is needed because
            traits in PhenomeXcan are mapped to EFO and then to DOID, and sometimes
            one trait maps to several DOIDs. This list helps to select among those, and
            it should contains the DOID present in the gold standard.
        force_run:
            If results already exist, it halt the execution.
        n_top_conditions:
            Number of conditions (genes or modules/LVs) to use to compute
            predictions. If None, it uses all. If it is a number then it ranks
            conditions and takes the top only.
        use_abs:
            Related to `n_top_conditions`. If True, then when ranking conditions
            (genes/LVs) it considers the absolute values.

    Returns:
        None
    """
    # prepare output file name suffix
    output_file_suffix = "all_genes"
    if n_top_conditions is not None:
        output_file_suffix = f"top_{n_top_conditions}_genes"

    print(f"predicting {output_file_suffix}...", end="")

    # create output dir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = Path(
        output_dir,
        f"{gene_trait_data_filename.stem}-{output_file_suffix}-prediction_scores.h5",
    ).resolve()

    if output_file.exists() and not force_run:
        print("  already run")
        return

    print("")

    if n_top_conditions is not None:
        # if n_top_conditions is given, then for each trait (columns) zero all
        # genes/LVs that are not among the top ones. Keep in mind that this only
        # works for dot product type of prediction.
        gene_trait_data = gene_trait_data.apply(
            lambda x: _zero_nontop_genes(x, n_top_conditions, use_abs)
        )

    # compute predictions
    drug_disease_assocs = prediction_function(drug_gene_data, gene_trait_data)
    print(f"  shape: {drug_disease_assocs.shape}")

    print(f"  saving to: {str(output_file)}")
    with pd.HDFStore(output_file, mode="w", complevel=4) as store:
        # save prediction across all traits
        print(f"    saving full predictions...")
        _save_predictions(drug_disease_assocs, store, "full_prediction")

        # save prediction for comparison with gold-standard
        print(f"    saving classifier data...")
        drug_disease_assocs = Trait.map_to_doid(
            drug_disease_assocs, preferred_doid_list, combine="max"
        )
        print(f"      shape (after DOID map): {drug_disease_assocs.shape}")
        assert drug_disease_assocs.index.is_unique
        assert drug_disease_assocs.columns.is_unique

        _save_predictions(drug_disease_assocs, store, "prediction")

        # save metadata
        metadata = pd.DataFrame(
            {
                "method": [base_method_name],
                # FIXME: consider adding a categorical/string value for
                #  "n_top_genes", instead of numerical. See discussion here:
                #  https://github.com/greenelab/phenoplier/pull/35#discussion_r619476007
                "n_top_genes": [-1.0 if n_top_conditions is None else n_top_conditions],
                "data": [gene_trait_data_filename.stem],
            }
        )
        store.put("metadata", metadata, format="table")


def _save_predictions(drug_disease_assocs, store, key_name):
    """
    Saves the predictions into an HDFStore using the key_name as key.
    """
    predictions_data = (
        drug_disease_assocs.unstack()
        .reset_index()
        .rename(columns={"level_0": "trait", "perturbagen": "drug", 0: "score"})
    )

    predictions_data["trait"] = predictions_data["trait"].astype("category")
    predictions_data["drug"] = predictions_data["drug"].astype("category")

    assert predictions_data.shape == predictions_data.dropna().shape
    print(f"      shape: {predictions_data.shape}")
    display(predictions_data.describe())

    # save
    print(f"      key: {key_name}")

    store.put(key_name, predictions_data, format="table")


def predict_dotprod_neg(
    drug_gene_data,
    gene_trait_data_filename,
    gene_trait_data,
    output_dir_base,
    base_method_name,
    preferred_doid_list,
    force_run,
    n_top_conditions=None,
    use_abs=False,
):
    """
    Predicts drug-disease associations using the dot product between
    drug_gene_data and gene_trait_data. The arguments and return are the same as
    in _predict function.

    This function computes the dot product between each trait vector and drug
    vector, multiplying the standardized effect size of all genes (for the
    trait) and the expression profiles of all genes (for the drug). The result
    is multiplied by -1, since we want a higher/positive score when the signs of
    these two types of data are different for a particular gene: if a higher
    expression of a gene is associated with a disease, then the standardized
    effect size is positive; a drug that decreases the expression of that gene
    will have a negative z-score.
    """
    output_dir = output_dir_base / "dotprod_neg"

    def _func(drugs_data, gene_assoc_data):
        return -1.0 * drugs_data.T.dot(gene_assoc_data)

    _predict(
        drug_gene_data,
        gene_trait_data_filename,
        gene_trait_data,
        output_dir,
        _func,
        base_method_name,
        preferred_doid_list,
        force_run,
        n_top_conditions,
        use_abs,
    )
