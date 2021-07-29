import warnings
import ast
from functools import lru_cache
from pathlib import Path
import pandas as pd

from entity import Gene
from data.cache import read_data
import conf


class ExperimentDataReader(object):
    """
    It downloads and process a gene expression study (SRP) from recount2
    given the SRP accession. It allows to extract properties from the
    "characteristics" column.

    The data that gets downloaded from recount2 (a tsv file) is the same the
    user gets when (in the current shinyapp of recount2) he/she clicks the
    `link` button under the `phenotype` column.

    Args:
        srp_code: an SRP code.
        srp_dir: a directory where data is downloaded to.
        compact: if True, only three columns are kept for each tsv file
            downloaded: "project", "run", and "characteristics".
    """

    UNKNOWN_ATTRIBUTE_COLUMN_NAME = "[PHENOPLIER] Unknown attribute"

    def __init__(self, srp_code: str, srp_dir: str, compact: bool = False):
        self.srp_code = srp_code
        self.srp_dir = Path(srp_dir).resolve()
        self.compact = compact

        self.srp_data_file = self.srp_dir / f"{self.srp_code}.tsv"
        self.characteristics_column_names = []

    @property
    @lru_cache(maxsize=None)
    def data(self) -> pd.DataFrame:
        """
        It downloads and process the SRP data, returning a dataframe.

        Returns:
            A dataframe with all the experiments for this study in the rows, and
            the metadata in the columns. The "characteristics" column (which
            contains values such as "c('key': 'value', 'key2': 'value2')",
            although quoting and other format parameters sometimes differ) is
            split into more columns (such as "key" and "key2" and the
            corresponding values).
        """

        self._download_srp_file()

        df = pd.read_csv(
            self.srp_data_file,
            sep="\t",
            usecols=["project", "run", "characteristics"] if self.compact else None,
        )

        df_with_characteristics = df.apply(self._process_characteristics, axis=1)

        # get the new columns added by self._process_characteristics
        self.characteristics_column_names = df_with_characteristics.columns.difference(
            df.columns
        )

        return df_with_characteristics

    @staticmethod
    def _process_characteristics(row: pd.Series) -> pd.Series:
        """
        This method is intended to be run for each row of a dataframe. It reads
        the 'characteristics' column, taking each key/value pair, and stores it
        as a new column.

        Args:
            row: a row of an SRP accession data from recount2.

        Returns:
            A new series object with new columns from the 'characteristics'
            original column.
        """
        # deal with different formats in the "characteristics" column
        if row.characteristics.startswith("c("):
            # R vectors style
            chars = ast.literal_eval(row.characteristics[1:])
        else:
            if row.characteristics.count(":") == 1:
                chars = (row.characteristics,)
            else:
                chars = (
                    f"{ExperimentDataReader.UNKNOWN_ATTRIBUTE_COLUMN_NAME}:{row.characteristics}",
                )

        for c in chars:
            key, value = c.split(":", 1)
            row[key.strip()] = value.strip()

        return row

    def _download_srp_file(self):
        """
        It downloads the metadata for an SRP accession from recount2. This is
        equivalent to clicking on the "link" button/link (under the "phenotype"
        column) for a particular SRP accession in the recount2 shinyapp
        (https://jhubiostatistics.shinyapps.io/recount/).

        Returns:
            None
        """
        # if the file was already downloaded, quit
        if self.srp_data_file.exists() and self.srp_data_file.stat().st_size > 0:
            return

        self.srp_dir.mkdir(exist_ok=True)

        import requests

        # download the file
        download_link = (
            f"http://duffel.rail.bio/recount/{self.srp_code}/{self.srp_code}.tsv"
        )
        file_content = requests.get(download_link, allow_redirects=True)
        with open(self.srp_data_file, "wb") as fh:
            fh.write(file_content.content)

        # make sure the file was successfully downloaded and has some content
        assert (
            self.srp_data_file.exists() and self.srp_data_file.stat().st_size > 0
        ), "File could not be downloaded"


class LVAnalysis(object):
    """
    It allows to perform some analyses on an LV (gene module) from MultiPLIER,
    in addition to conveniently retrieve the list of top genes, traits and
    conditions associated with an LV.

    Args:
        lv_name: the name or identifier of the LV, which needs to have the
            format "LV{number}".

        lvs_traits_data: a dataframe with the projection of S-MultiXcan results
            into the recount2 latent space (MultiPLIER). It must have traits in
            the index (rows) and LVs in the columns.
    """

    RECOUNT2_SRP_DIR = Path(conf.RECOUNT2["BASE_DIR"], "srp").resolve()

    def __init__(self, lv_name: str, lvs_traits_data: pd.DataFrame = None):
        self.lv_name = lv_name
        self.lv_number = int(lv_name[2:])

        # these data is read here and then discarded (it is not kept inside the
        # instance)
        multiplier_model_z = read_data(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])
        multiplier_model_b = read_data(conf.MULTIPLIER["MODEL_B_MATRIX_FILE"])

        self._init_genes(multiplier_model_z)
        self._init_conditions(multiplier_model_b)

        if lvs_traits_data is not None:
            self._init_traits(lvs_traits_data)

        self._experiment_attrs_sim = {}

    @staticmethod
    def _assign_gene_band(gene_name: str):
        """
        Given a gene symbol, it gets its cytoband.

        Args:
            gene_name: a gene symbol.

        Returns:
            The gene's cytoband.
        """
        if gene_name in Gene.GENE_NAME_TO_ID_MAP:
            return Gene(name=gene_name).band

        return None

    def _init_genes(self, multiplier_model_z: pd.DataFrame):
        """
        It initializes a dataframe with genes (and metadata) for this LV.

        Args:
            multiplier_model_z: the gene loadings from MultiPLIER (matrix z).

        Returns:
            None
        """
        self.lv_genes = multiplier_model_z[self.lv_name].sort_values(ascending=False)
        self.lv_genes = self.lv_genes.reset_index().rename(
            columns={"index": "gene_name"}
        )
        self.lv_genes = self.lv_genes.assign(
            gene_band=self.lv_genes["gene_name"]
            .apply(self._assign_gene_band)
            .astype("category")
        )

    def _init_conditions(self, multiplier_model_b):
        """
        It initializes a dataframe with the conditions associated with this LV.
        "Conditions" are the experiments (SRR) conducted in the studies (SRP) of
        recount2.

        Args:
            multiplier_model_b: the latent space from MultiPLIER (matrix b).

        Returns:
            None
        """
        # get the LV name used in the MultiPLIER data
        lv_name_on_b = multiplier_model_b.index[
            pd.Series(multiplier_model_b.index).str.contains(
                f"^{self.lv_number},|LV {self.lv_number}$", regex=True
            )
        ]
        assert len(lv_name_on_b) == 1
        lv_name_on_b = lv_name_on_b[0]

        # create the dataframe for the conditions of this LV: it has the project
        # (SRP), the experiment (SRR), the full id given by both, and the LV
        # value.
        lv_conds = multiplier_model_b.loc[lv_name_on_b].sort_values(ascending=False)
        lv_conds = lv_conds.to_frame(self.lv_name).reset_index()
        lv_conds = lv_conds.assign(
            project=lv_conds["index"].apply(lambda x: x.split(".")[0])
        )
        lv_conds = lv_conds.assign(
            experiment=lv_conds["index"].apply(lambda x: x.split(".")[1])
        )
        self.lv_conds = lv_conds.rename(columns={"index": "experiment_id"})

    def _init_traits(self, lvs_traits_data):
        """
        It initializes a series with the traits scores for this LV.

        Args:
            lvs_traits_data: a dataframe with the projection of S-MultiXcan
                results into the recount2 latent space (MultiPLIER). It should have
                traits in the index (rows) and LVs in the columns.

        Returns:
            None
        """
        self.lv_traits = lvs_traits_data[self.lv_name].sort_values(ascending=False)

    @lru_cache(maxsize=None)
    def get_top_projects(self, quantile: float):
        """
        For each project (SRP), it takes the maximum LV value across all experiments
        and sorts SRP by that. Then it returns a list of those having a value greater
        than the specified quantile.

        Args:
            quantile: a quantile that will be obtained from the list of maximum
            values across all experiments of each SRP

        Returns:
            A list of top SRP codes associated with the LV.
        """
        df_lv_maxs = self.lv_conds.set_index("project")[self.lv_name].sort_values(
            ascending=False
        )

        quantile_value = self.lv_conds[self.lv_name].quantile(quantile)

        top_projects = (
            df_lv_maxs.reset_index()
            .groupby("project")
            .max()
            .squeeze()
            .sort_values(ascending=False)
        )
        top_projects = top_projects[top_projects > quantile_value]

        return top_projects.index

    @lru_cache(maxsize=None)
    def get_experiments_data(self, quantile=0.99, debug=True, show_warnings=True):
        """
        For each top SRP code associated with this LV (see method get_top_projects),
        it obtains all experiments and concatenates everything into one dataframe.

        Args:
            quantile: quantile parameter of method get_top_projects
            debug: activates debugging messages
            show_warnings: activates warnings messages

        Returns:

        """
        experiments_df_list = []
        dfs_lengths = 0
        new_chars_columns = set()

        top_projects = self.get_top_projects(quantile)
        for srp_code in top_projects:
            if debug:
                print(srp_code, end=", ", flush=True)

            edr = ExperimentDataReader(
                srp_code, srp_dir=LVAnalysis.RECOUNT2_SRP_DIR, compact=True
            )

            try:
                dfs_lengths += edr.data.shape[0]
            except SyntaxError:
                # ignore syntax errors (raised by ast because of malformed
                # values in the "characteristics" column)
                continue

            new_chars_columns.update(edr.characteristics_column_names)

            experiments_df_list.append(edr.data)

        if show_warnings and (len(experiments_df_list) != top_projects.shape[0]):
            warnings.warn(
                f"Not all experiments data could be loaded "
                f"({len(experiments_df_list)} != {top_projects.shape[0]})"
            )

        df = pd.concat(experiments_df_list, ignore_index=True)

        # some testing
        assert df.shape[0] == dfs_lengths
        assert all([c in df.columns for c in new_chars_columns])

        # add LV value
        df = (
            df.set_index(["project", "run"])
            .assign(
                **{
                    self.lv_name: self.lv_conds[
                        self.lv_conds["project"].isin(top_projects)
                    ].set_index(["project", "experiment"])[self.lv_name]
                }
            )
            .dropna(subset=[self.lv_name])
            .drop("characteristics", axis=1)
        )

        return df

    def get_attributes_variation_score(self, func: str = "var") -> pd.Series:
        """
        For each column in the experiments dataframe (see method
        get_experiments_data), it computes a statistic (such as the variance)
        and returns a list column:statistic_value.

        This method is used to rank the attributes present across different
        experiments (examples of attributes are "cell type", "tissue",
        "disease", etc).

        Args:
            func: a statistic to be computed on each column of the data. It can
                be any function present in a pandas.DataFrame. By default, it is
                "var" (variance).

        Returns:
            A sorted series with column names (attributes of data) in index and
                their statistics values.
        """
        data = self.get_experiments_data()

        values = {}
        for col in data.columns.drop(self.lv_name):
            _tmp = data[[col, self.lv_name]].dropna()
            _tmp = getattr(_tmp, func)()
            values[col] = _tmp[self.lv_name]

        return pd.Series(values).sort_values(ascending=False)

    def plot_attribute(
        self,
        imp_f: str,
        linewidth: float = 1,
        hue: str = None,
        study_id: str = None,
        top_x_values: int = None,
        ax=None,
    ):
        """
        It creates a boxplot with the LV values for all the categories for a one
        attribute of the data (such as "cell type"). For instance, if imp_f is
        "tissue", this plot would have values such as "Heart", "Lung", etc in
        the x-axis, and the LV value in the y-axis. Values in the x-axis are
        sorted by their median.

        This function is not used for final figure production, but just as a
        quick way to explore relationships. For example, LV's genes could be
        related to cardiovascular traits, and all experiments performed in
        cardiomyocytes (one value of the x-axis) might have a higher value than
        the rest on average.

        Args:
            imp_f: attribute to plot, such as "cell type" or "tissue"
            linewidth: same as in seaborn.boxplot function.
            hue: same as in seaborn.boxplot function.
            study_id: SRP code to include (all the rest will be ignored). It is
                useful when you want to explore patterns in a single project.
            top_x_values: top attribute categories to include. By default all
                are included.
            ax: matplotlib Axes, same as in seaborn.boxplot function.

        Returns:
            A matplotlib Axes instance.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = self.get_experiments_data()

        title = ""

        if study_id is not None:
            data = data.loc[study_id]
            title += f"[Samples from {study_id} only]"

        features = [imp_f, self.lv_name]
        if hue is not None:
            features.append(hue)

        data = data[features].dropna()

        cat_order = data.groupby(imp_f).median().squeeze()
        if isinstance(cat_order, float):
            warnings.warn(f"Single value for {imp_f}: {cat_order}")
            return

        cat_order = cat_order.sort_values(ascending=False)
        if top_x_values is not None:
            cat_order = cat_order.head(top_x_values)

        cat_order = cat_order.index

        ax = sns.boxplot(
            data=data,
            x=imp_f,
            y=self.lv_name,
            order=cat_order,
            hue=hue,
            linewidth=linewidth,
            ax=ax,
        )
        plt.xticks(rotation=45, horizontalalignment="right")
        plt.title(title)
        return ax
