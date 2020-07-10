"""
Classes and code with MultiPLIER related functionality.
"""
from pathlib import Path

import conf
import numpy as np
import pandas as pd


class MultiplierProjection(object):
    """Projects new data into the MultiPLIER latent space."""
    def __init__(self):
        pass

    def transform(self, y, multiplier_compatible=True):
        """Projects a gene dataset into the MultiPLIER model.

        This code is a reimplementation in Python of the function GetNewDataB
        (https://github.com/greenelab/multi-plier/blob/v0.2.0/util/plier_util.R),
        more suitable and convenient for the PhenoPLIER project (almost entirely
        written in Python).

        It basically row-normalizes (z-score) the given dataset, keeps only the genes
        in common with the MultiPLIER model, and adds the missing ones as zeros (mean).

        Args:
            y (pandas.DataFrame): the new data to be projected. Gene symbols are
                expected in rows. The columns could be conditions/samples, but in the
                PhenoPLIER context they could also be traits/diseases or perturbations
                (Connectivity Map).
            multiplier_compatible (bool): if True, it will try to be fully compatible
                with the GetNewDataB function in some situations (for instance, if the
                new data contains NaNs).

        Returns:
            A pandas.DataFrame with the projection of the input data into the MultiPLIER
            latent space. The latent variables of the MultiPLIER model are in the rows,
            and the columns of the input data (conditions, traits, drugs, etc).
        """
        z = self._read_model_z()
        metadata = self._read_model_metadata()

        # nothing special is done if the input data contains NaNs, but it will raise
        # a warning for the user.
        if y.isna().any().any():
            import warnings
            warnings.warn('Input data contains NaN values.')

            # if multiplier_compatible, just mimic the same behavior of function
            # GetNewDataB and return a DataFrame of NaN values.
            if multiplier_compatible:
                return pd.DataFrame(
                    data=np.nan,
                    index=z.columns.copy(),
                    columns=y.columns.copy(),
                )

        # row-standardize the data with z-score
        y_std = y \
            .sub(y.mean(1), axis=0) \
            .div(y.std(1), axis=0)

        model_genes = z.index
        data_genes = y_std.index
        common_genes = model_genes.intersection(data_genes)

        # select from input data only genes in common with model, and add missing ones
        # as zeros (mean).
        y_std = y_std\
            .loc[common_genes]\
            .append(
                pd.DataFrame(
                    0,
                    index=model_genes.difference(data_genes),
                    columns=y_std.columns.copy(),
                )
            )

        # get the precision matrix of gene loadings (z matrix)
        z_cov_inv = pd.DataFrame(
            data=np.linalg.pinv(
                z.T.dot(z) +
                metadata['L2'] * np.identity(z.shape[1])
            ),
            index=z.columns.copy(),
            columns=z.columns.copy(),
        )

        # perform final matrix multiplication: (z^T z + l2 I)^{-1} z^T Y
        return (
            z_cov_inv
                .dot(z.T)
                .dot(y_std)
        )

    @staticmethod
    def _read_multiplier_file(filename):
        """Reads a pickle file located in the MultiPLIER base dir."""
        input_file = Path(
            conf.MULTIPLIER['BASE_DIR'],
            filename
        ).resolve()

        return pd.read_pickle(input_file)

    @staticmethod
    def _read_model_z():
        """Returns the MultiPLIER Z matrix (gene loadings)."""
        return MultiplierProjection._read_multiplier_file('multiplier_model_z.pkl')

    @staticmethod
    def _read_model_metadata():
        """Returns metadata of the MultiPLIER model."""
        return MultiplierProjection._read_multiplier_file('multiplier_model_metadata.pkl')
