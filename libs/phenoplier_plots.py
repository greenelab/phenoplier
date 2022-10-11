import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples


def qqplot(observed_pvalues, expected_pvalues=None):
    """
    A QQ plot given an array of observed pvalues and expected pvalues.
    If expected pvalues are not given, then a uniform distribution of -log10
    pvalues are generated instead.
    """
    obs_data = observed_pvalues
    if hasattr(observed_pvalues, "to_numpy"):
        obs_data = observed_pvalues.to_numpy()

    n = obs_data.shape[0]
    observed_data = -np.log10(obs_data)

    if expected_pvalues is not None:
        exp_data = expected_pvalues
        if hasattr(expected_pvalues, "to_numpy"):
            exp_data = expected_pvalues.to_numpy()

        expected_data = -np.log10(exp_data)
    else:
        uniform_data = np.array([i / (n + 1) for i in range(1, n + 1)])
        expected_data = -np.log10(uniform_data)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    fig = qqplot_2samples(expected_data, observed_data, line="45", ax=ax)

    ax.set_xlim(expected_data.min() - 0.05, expected_data.max() + 0.05)

    ax.set_xlabel("$-\log_{10}$" + "(expected pvalue)")
    ax.set_ylabel("$-\log_{10}$" + "(observed pvalue)")

    ax.set_title(f"Number of data points:  {n}")

    return fig, ax
