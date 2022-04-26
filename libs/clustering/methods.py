import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree


class ClusterInterpreter(object):
    """
    It helps to interpret clustering results by finding the top latent variables (LV)
    that discriminate members of a cluster.

    Given the original data, a partition of its traits, and a cluster id, it
    trains a Decision Tree classifier with two classes: the traits that belong
    to cluster id and all the rest. The idea is to get the features/LVs that
    discriminate the cluster members from all the other traits.

    Args:
        threshold: threshold to be used to select features/LVs from the root
            node of a tree (default: 0). For this case, it makes sense to use always
            a greater-or-equeal to zero value here (since LVs with negative values
            are meaningless).
        max_features: maximum features to be selected
        max_features_to_explore: maximum number of features to explore
        decision_tree_args: arguments given to the DecisionTreeClassifier
            instance.
    """

    def __init__(
        self,
        threshold: float = 0,
        max_features: int = 10,
        max_features_to_explore: int = 50,
        decision_tree_args: dict = None,
    ):
        self.threshold = threshold
        self.max_features = max_features
        self.max_features_to_explore = max_features_to_explore

        self.decision_tree_args = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": 1,
            "max_features": None,
            "random_state": None,
        }
        if decision_tree_args is not None:
            self.decision_tree_args.update(decision_tree_args)

        self.features_ = None

    def fit(self, data: pd.DataFrame, partition: np.ndarray, cluster_id: int):
        """
        Fits the cluster interpreter with the given data, a partition of it, and
        a cluster label (which must be present in the partition). It repeatedly
        trains a Decision Tree classifier on the data and an "artificial" set of
        labels containing the traits with cluster_id as positive cases and the
        rest as negative ones. Then, it picks the feature from the root node
        (provided the threshold is greater than self.threshold), and starts all
        over again by removing the previously selected feature from the data. It
        stops when the maximum number of features is reached, or when the
        maximum number of features to explored is reached, whatever happens
        first.

        When it finishes, the attribute self.features_ is created with a list
        of selected features.

        Args:
            data: a dataframe with traits in rows and features/LVs in columns.
            partition: a partition of the data (the result of a clustering
                algorithm applied on data).
            cluster_id: a cluster id present in partition.

        Returns:
            self
        """
        assert partition.shape[0] == data.shape[0], (
            f"Partition shape does not match number of rows in data "
            f"({partition.shape[0]} != {data.shape[0]})"
        )

        assert cluster_id in np.unique(
            partition
        ), f"Cluster id does not exist in partition ({cluster_id})"

        # create binary partition to compare cluster members with the rest
        part = np.zeros_like(partition)
        part[partition == cluster_id] = 1

        i = 0
        _features_selected = []
        _new_data = data
        _next_feature_drop = []
        while (
            _new_data.shape[1] > 0
            and i < self.max_features_to_explore
            and len(_features_selected) < self.max_features
        ):
            # remove the root node from the previously trained tree
            _new_data = _new_data.drop(columns=_next_feature_drop)

            # TODO: it would probably be much better/efficient to use the
            # feature splitter that the decision tree is using internally
            # instead of training a new classifier every time
            clf = DecisionTreeClassifier(**self.decision_tree_args)
            clf.fit(_new_data, part)

            root_node_feature = {}
            feature_idx = clf.tree_.feature[0]
            root_node_feature["name"] = _new_data.columns[feature_idx]
            root_node_feature["idx"] = int(root_node_feature["name"][2:])
            root_node_feature["threshold"] = clf.tree_.threshold[0]
            root_node_feature["impurity"] = clf.tree_.impurity[0]
            root_node_feature["value"] = clf.tree_.value[0]
            root_node_feature["n_samples"] = clf.tree_.n_node_samples[0]

            _next_feature_drop = [root_node_feature["name"]]

            # do not continue if feature value is less than specified threshold
            if root_node_feature["threshold"] < self.threshold:
                continue

            _features_selected.append(pd.Series(root_node_feature))

            i = i + 1

        self.features_ = (
            pd.DataFrame(_features_selected)
            .set_index("idx")
            .sort_values("threshold", ascending=False)
        )

        return self


class DeltaSpectralClustering(SpectralClustering):
    """
    It extends SpectralClustering by accepting a distance matrix as input and
    applying a Gaussian kernel before fitting (this is suggested in the
    sklearn documentation of :obj:`sklearn.cluster.SpectralClustering` to
    convert a distance matrix to a similarity matrix suitable for this
    clustering algorithm).
    """

    def __init__(self, delta=1.0, **kwargs):
        """
        The same arguments of :obj:`sklearn.cluster.SpectralClustering`, plus
        a ``delta`` parameter. If ``affinity`` is provided, it will be overridden
        to "precomputed".

        Args:
            delta: it represents the width of the Gaussian kernel applied to the
                input distance matrix before fitting.
            **kwargs: any parameter accepted by
                :obj:`sklearn.cluster.SpectralClustering`. If ``affinity`` is
                provided, a warning is raised and it is overridden by
                "precomputed".
        """

        if "affinity" in kwargs:
            warnings.warn(
                "The 'affinity' parameter was specified; it will be overridden "
                + "with 'precomputed'"
            )

        kwargs["affinity"] = "precomputed"

        super().__init__(**kwargs)

        self.delta = delta

    def fit(self, X, y=None):
        """
        Performs spectral clustering from an affinity matrix derived from the
        distance matrix provided. It applies a Gaussian kernel before fitting::

            np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

        where the ``delta`` represents the width of the Gaussian kernel.

        Args:
            X: array-like, shape (n_samples, n_samples)
                Distance matrix for all instances to cluster (0 means identical
                instances, and high values mean very dissimilar instances).
            y: Ignored
                Not used, present here for API consistency by convention.

        Returns:
            self
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                "The data has to be a squared distance matrix;"
                + f" current shape is {X.shape}"
            )

        # This Gaussian kernel is suggested in the sklearn documentation for
        # SpectralClustering. It converts a distance matrix to a similarity
        # matrix.
        X = np.exp(-(X**2) / (2.0 * self.delta**2))

        return super().fit(X, y)
