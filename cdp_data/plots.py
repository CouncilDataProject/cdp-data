#!/usr/bin/env python

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from . import keywords

if TYPE_CHECKING:
    from matplotlib.axes import SubplotBase

###############################################################################


def set_cdp_plotting_styles() -> None:
    """Set the standard CDP plotting styles."""
    sns.set_context("paper")
    sns.set_theme(
        style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False}
    )
    sns.set_style("darkgrid", {"grid.color": "#000000", "grid.linestyle": ":"})
    sns.set_palette(
        [
            "#ff6a75",  # cherry-red
            "#0060df",  # ocean-blue
            "#068989",  # moss-green
            "#712290",  # purple
            "#FFA537",  # orange
            "#FF2A8A",  # pink
            "#9059FF",  # lavender
            "#00B3F5",  # light-blue / sky-blue
            "#005e5e",  # dark blueish green
            "#C50143",  # dark-red / maroon
            "#3fe1b0",  # seafoam / mint
            "#063F96",  # dark-blue / navy-blue
            "#FFD567",  # banana-yellow
        ]
    )


def _recurse_axes_grid_to_fix_datetimes(
    arr_or_subplot: Union[np.ndarray, "SubplotBase"],
) -> None:
    if isinstance(arr_or_subplot, np.ndarray):
        for item in arr_or_subplot:
            _recurse_axes_grid_to_fix_datetimes(item)
    else:
        ax = arr_or_subplot
        xticks = ax.get_xticks()
        xticks_dates = [datetime.fromtimestamp(x).strftime("%b %Y") for x in xticks]
        ax.set_xticklabels(xticks_dates)
        ax.tick_params(axis="x", rotation=50)


def _prepare_ngram_history_plotting_data(
    data: pd.DataFrame,
    ngram_col: str,
    value_col: str,
    dt_col: str,
    keep_cols: Union[List[str], None] = None,
) -> pd.DataFrame:
    """
    A utility function to prepare ngram history data for plotting.

    Parameters
    ----------
    data: pd.DataFrame
        The data to prepare for plotting.
    ngram_col: str
        The column name for which the "ngram" is stored.
        Generally this is the column for which plots are split to small multiples.
        For example, a single plot for "police", "housing", etc.
    value_col: str
        The column name for which the value of each ngram is stored.
    dt_col: str
        The column name for which the datetime is stored.
    keep_cols: List[str]
        Any extra columns to keep.

    Returns
    -------
    plot_data: pd.DataFrame
        The grouped, sorted, and datetime formatted data ready for plotting.

    See Also
    --------
    prepare_ngram_relevancy_history_plotting_data
        Function to prepare specifically ngram relevancy history data for plotting.
    prepare_ngram_usage_history_plotting_data
        Function to prepare specifically ngram usage history data for plotting.
    """
    # Handle empty keep cols
    if not keep_cols:
        keep_cols = []

    # Select down to just the columns we want
    # Reset index
    # Sort values by datetime
    subset = (
        data[[ngram_col, value_col, dt_col, *keep_cols]]
        .sort_values([ngram_col, dt_col])
        .reset_index(drop=True)
    )

    # Ensure the date col is a datetime / pd.Timestamp
    subset[dt_col] = pd.to_datetime(subset[dt_col])

    # Also create a column of the timestamp value
    subset["timestamp_posix"] = subset[dt_col].apply(
        lambda timestamp: timestamp.timestamp()
    )

    return subset


def prepare_ngram_relevancy_history_plotting_data(
    data: pd.DataFrame,
    ngram_col: str = "query_gram",
    value_col: str = "value",
    dt_col: str = "event_datetime",
) -> pd.DataFrame:
    """
    Prepare an ngram relevancy history DataFrame specifically for plotting.
    This function will subset the DataFrame to just the provided columns and
    will only store a single value for each day if there are multiple
    (keeping the max value).

    Parameters
    ----------
    data: pd.DataFrame
        The data to prepare for plotting.
    ngram_col: str
        The column name for which the "ngram" is stored.
        Default: "query_gram"
    value_col: str
        The column name for which the value of each ngram is stored.
        Default: "value"
    dt_col: str
        The column name for which the datetime is stored.
        Default: "event_datetime"

    Returns
    -------
    prepared: pd.DataFrame
        The subset and max selected dataset reading for plotting.

    See Also
    --------
    cdp_data.keywords.get_ngram_relevancy_history
        The dataset retrival function which should generally paired with this function.
    """
    # Basic preparation
    data = _prepare_ngram_history_plotting_data(
        data=data,
        ngram_col=ngram_col,
        value_col=value_col,
        dt_col=dt_col,
    )

    # Keep max for date
    data = (
        data.groupby([ngram_col, pd.Grouper(key=dt_col, freq="D")]).max().reset_index()
    ).replace([None])

    return data


def _prepare_ngram_usage_history_plotting_data(
    ngram: str,
    data: pd.DataFrame,
    strict: bool = False,
    ngram_col: str = "ngram",
    percent_col: str = "day_ngram_percent_usage",
    dt_col: str = "session_date",
) -> pd.DataFrame:
    """
    Prepare an ngram usage history DataFrame specifically for plotting.
    This function will stem and clean the provided ngram, subset to just the data
    of interest, and prepare the rest of the data for plotting.

    Parameters
    ----------
    ngram: str
        A single ngram of interest to plot.
    data: pd.DataFrame
        The data to prepare for plotting.
    strict: bool
        Should the provided ngram be stemmed or left unstemmed for a more
        strict usage history.
        Default: False (stem and clean the provided ngram)
    ngram_col: str
        The column name for which the "ngram" is stored.
        Default: "ngram"
    percent_col: str
        The column name for which the percent usage of each ngram is stored.
        Default: "day_ngram_percent_usage"
    dt_col: str
        The column name for which the date is stored.
        Default: "session_date"

    Returns
    -------
    prepared: pd.DataFrame
        The subset and prepared for plotting dataset.

    See Also
    --------
    cdp_data.keywords.compute_ngram_usage_history
        The dataset loading function which should generally paired with this function.
    """
    # Prepare ngram for user
    if not strict:
        ngram = keywords._stem_n_gram(ngram)

    # Select down to just the ngram we want
    subset = data.loc[data[ngram_col] == ngram]

    # Check subset length for better error
    if len(subset) == 0:
        raise ValueError(
            f"Provided (or stemmed) ngram ('{ngram}') resulted in "
            f"zero matching rows for plotting."
        )

    return _prepare_ngram_history_plotting_data(
        data=subset,
        ngram_col=ngram_col,
        value_col=percent_col,
        dt_col=dt_col,
        keep_cols=["infrastructure"],
    )


def plot_ngram_usage_histories(
    ngram: Union[str, List[str]],
    gram_usage: pd.DataFrame,
    strict: bool = False,
    lmplot_kws: Union[Dict[str, Any], None] = None,
    tqdm_kws: Union[Dict[str, Any], None] = None,
) -> "sns.FacetGrid":
    """
    Select and plot specific ngram usage histories from the provided gram usage
    DataFrame.

    Parameters
    ----------
    ngram: Union[str, List[str]]
        The unigrams, bigrams, or trigrams to retrieve history for.
        Note: Must provide all unigrams, bigrams, or trigrams, cannot mix gram size, and
        the gram size should be the same as the grams stored in the provided gram_usage
        DataFrame.
    gram_usage: pd.DataFrame
        A pandas DataFrame of all found ngrams (stemmed and cleaned or unstemmed and
        uncleaned) from the data and their counts for each session and their percentage
        of use as a percent of their use for the day over the sum of all other ngrams
        used that day.
    strict: bool
        Should all ngrams be stemmed or left unstemmed for a more strict usage history.
        Default: False (stem and clean all grams in the dataset)
    lmplot_kws: Dict[str, Any]
        Any extra kwargs to provide to sns.lmplot.
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

    Returns
    -------
    grid: sns.FacetGrid
        The small multiples FacetGrid of all ngrams and infrastructures found in the
        provided dataset.

    See Also
    --------
    cdp_data.keywords.compute_ngram_usage_history
        Function to generate ngram usage history DataFrame.
    """
    # Handle default dicts
    if not lmplot_kws:
        lmplot_kws = {}
    if not tqdm_kws:
        tqdm_kws = {}

    # Always cast ngram to list for easier API
    if isinstance(ngram, str):
        ngram = [ngram]

    # TODO:
    # Assert all ngrams are the same size

    # TODO:
    # Add keyword grouping??

    # Store prepared subsets
    gram_histories = []

    # Process the grams for the infrastructure
    for gram in tqdm(
        ngram,
        desc="Preparing plotting data for each ngram",
        **tqdm_kws,
    ):
        gram_history = _prepare_ngram_usage_history_plotting_data(
            gram,
            data=gram_usage,
            strict=strict,
        )

        # Attach this gram history to all
        gram_histories.append(gram_history)

    # Convert histories to dataframe
    gram_histories = pd.concat(gram_histories)

    # Plot all the data
    grid = sns.lmplot(
        x="timestamp_posix",
        y="day_ngram_percent_usage",
        data=gram_histories,
        **lmplot_kws,
    )

    # Fix the axes to actual date formats
    _recurse_axes_grid_to_fix_datetimes(grid.axes)

    # Set axis labels
    grid.set_axis_labels("Date", "Ngram Usage (percent)")
    grid.tight_layout()

    return grid


def plot_query_semantic_similarity_history(
    semantic_history: pd.DataFrame,
    poolings: Iterable[str] = ("min", "max", "mean"),
    lmplot_kws: Union[Dict[str, Any], None] = None,
) -> "sns.FacetGrid":
    """
    Plot pre-computed semantic similarity history data.

    Parameters
    ----------
    semantic_history: pd.DataFrame
        The pre-computed semantic similarity history data.
    poolings: List[str]:
        Which poolings to plot (min, max, and/or mean).
        Default: ["min", "max", "mean"] (plot all)
    lmplot_kws: Dict[str, Any]:
        Extra keyword arguments to be passed to seaborn lmplot.

    Returns
    -------
    seaborn.FacetGrid
        The semantic similarity history plot.

    See Also
    --------
    cdp_data.keywords.compute_query_semantic_similarity_history
        The function used to generate the semantic_history data.
    """
    # Handle default dict
    if not lmplot_kws:
        lmplot_kws = {}

    # Add datetime timestamp
    semantic_history["session_date"] = pd.to_datetime(semantic_history["session_date"])
    semantic_history["timestamp_posix"] = semantic_history["session_date"].apply(
        lambda timestamp: timestamp.timestamp()
    )

    # Select poolings
    subsets: List[pd.DataFrame] = []
    for pooling in poolings:
        subsets.append(
            pd.DataFrame(
                {
                    "timestamp_posix": semantic_history["timestamp_posix"],
                    "query": semantic_history["query"],
                    "infrastructure": semantic_history["infrastructure"],
                    "value": semantic_history[f"day_similarity_{pooling}"],
                    "pooling": pooling,
                }
            )
        )
    render_ready = pd.concat(subsets)

    # Plot all the data
    grid = sns.lmplot(
        x="timestamp_posix",
        y="value",
        data=render_ready,
        **lmplot_kws,
    )

    # Fix the axes to actual date formats
    _recurse_axes_grid_to_fix_datetimes(grid.axes)

    # Set axis labels
    grid.set_axis_labels("Date", "Semantic Similarity")
    grid.tight_layout()

    return grid
