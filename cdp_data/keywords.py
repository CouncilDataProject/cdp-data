#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from cdp_backend.database import models as db_models
from cdp_backend.pipeline.transcript_model import Transcript
from cdp_backend.utils.string_utils import clean_text
from nltk import ngrams
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from . import datasets
from .utils import db_utils

if TYPE_CHECKING:
    import seaborn as sns

###############################################################################

# Logging
log = logging.getLogger(__name__)

###############################################################################
# Constants / Support Classes


@dataclass
class _NgramUsageProcessingParameters:
    session_id: str
    session_datetime: datetime
    transcript_path: Path


###############################################################################


def _stem_n_gram(n_gram: str) -> str:
    # Clean text
    n_gram = clean_text(n_gram, clean_stop_words=True, clean_emojis=True)

    # Raise error for no more text
    if len(n_gram) == 0:
        raise ValueError(f"Provided n_gram ({n_gram}) is empty after cleaning.")

    # Stem
    stemmer = SnowballStemmer("english")

    # Split and stem each
    return " ".join([stemmer.stem(span) for span in n_gram.split()])


def get_ngram_relevancy_history(
    ngram: str,
    strict: bool = False,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    infrastructure_slug: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull an n-gram's relevancy history from a CDP database.

    Parameters
    ----------
    ngram: str
        The unigram, bigram, or trigram to retrieve history for.
    strict: bool
        Should the provided ngram be used for a strict "unstemmed_gram" query or not.
        Default: False (stem and clean the ngram before querying)
    start_datetime: Optional[Union[str, datetime]]
        The earliest possible datetime for ngram history to be retrieved for.
        If provided as a string, the datetime should be in ISO format.
    end_datetime: Optional[Union[str, datetime]]
        The latest possible datetime for ngram history to be retrieved for.
        If provided as a string, the datetime should be in ISO format.
    infrastructure_slug: Optional[str]
        The optional CDP infrastructure slug to connect to.
        Default: None (you are managing the database connection yourself)

    Returns
    -------
    ngram_history: pd.DataFrame
        A pandas DataFrame of all IndexedEventGrams that match the provided ngram query
        (stemmed or unstemmed).

    See Also
    --------
    cdp_data.keywords.compute_ngram_usage_history
        Compute all ngrams usage history for a specific CDP session dataset.
        Useful for comparing how much discussion is comprised of specific ngrams.

    Notes
    -----
    This function pulls the TF-IDF (or other future indexed values) score for the
    provided ngram over time. This is a measure of relevancy to a document and not
    the same as Google's NGram Viewer which shows what percentage of literature
    used the term.
    """
    # Connect to infra
    if infrastructure_slug:
        db_utils.connect_to_database(infrastructure_slug)

    # Begin partial query
    keywords_collection = db_models.IndexedEventGram.collection

    # TODO:
    # Add datetime filtering
    if start_datetime or end_datetime:
        log.warning("`start_datetime` and `end_datetime` not implemented")

    # Determine strict query or stemmed query
    if strict:
        query = keywords_collection.filter(
            "unstemmed_gram",
            "==",
            ngram,
        )
    else:
        log.info("Parameter `strict` set to False, stemming query terms...")
        stemmed_gram = _stem_n_gram(ngram)
        query = keywords_collection.filter(
            "stemmed_gram",
            "==",
            stemmed_gram,
        )

    # Pull ngram history and cast to dataframe
    ngram_history = pd.DataFrame([d.to_dict() for d in query.fetch()])

    # Add column with original query
    ngram_history["query_gram"] = ngram

    # Get event details for each reference
    # Because we pulled from the index, there should never be duplicate
    # event calls so we are safe to just apply the whole df
    # Register `pandas.progress_apply` and `pandas.Series.map_apply`
    log.info("Attaching event metadata to each ngram history datum")
    ngram_history = db_utils.load_model_from_pd_columns(
        ngram_history,
        join_id_col="id",
        model_ref_col="event_ref",
    )

    # Unpack event cols
    ngram_history["event_datetime"] = ngram_history.apply(
        lambda row: row.event.event_datetime,
        axis=1,
    )

    return ngram_history


def _prepare_ngram_history_plotting_data(
    data: pd.DataFrame,
    ngram_col: str,
    value_col: str,
    dt_col: str,
    keep_cols: List[str] = [],
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


def fill_history_data_with_zeros(
    data: pd.DataFrame,
    ngram_col: str,
    dt_col: str,
) -> pd.DataFrame:
    """
    A utility function to fill ngram history data with zeros for all missing dates.

    Parameters
    ----------
    data: pd.DataFrame
        The ngram history data to fill dates for.
    ngram_col: str
        The column name for which the "ngram" is stored.
    dt_col: str
        The column name for which the datetime is stored.

    Returns
    -------
    data: pd.DataFrame
        A DataFrame filled with the original data and filled in with any missing dates
        with their values being set to zero.

    See Also
    --------
    prepare_ngram_history_plotting_data
        Subsets to plotting only columns and ensures values are sorted and grouped.
    """
    # Fill missing dates with zeros for each query gram
    def fill_dates(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(
            pd.date_range(
                df.index.min(),
                df.index.max(),
                name=dt_col,
            ),
            fill_value=0,
        )

    # Set index to the datetime column
    # Groupby the ngram col
    #     1. to make it so we don't lose the ngram col value
    #     2. to make it so we each ngram has their own complete datetime range
    # Apply the function to add missing dates
    # Drop the ngram col on the grouped data
    # Reset the index to ungroup the data by ngram (thus regaining the ngram column)
    return (
        data.set_index(dt_col)
        .groupby(ngram_col)
        .apply(fill_dates)
        .drop(ngram_col, axis=1)
        .reset_index()
    )


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
    get_ngram_relevancy_history
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


def _count_transcript_grams(
    processing_params: _NgramUsageProcessingParameters,
    ngram_size: int,
    strict: bool,
) -> pd.DataFrame:
    # Load transcript
    with open(processing_params.transcript_path, "r") as open_f:
        transcript = Transcript.from_json(open_f.read())

    # Start a counter
    counter: Counter = Counter()

    # For each sentence in transcript,
    # clean or not (based off strict),
    # count each cleaned or not gram
    for sentence in transcript.sentences:
        # Stopwords removed
        words = [
            clean_text(
                word.text,
                clean_stop_words=True,
                clean_emojis=True,
            )
            for word in sentence.words
        ]
        words = [word for word in words if len(word) > 0]

        # Create ngrams
        for gram in ngrams(words, ngram_size):
            if not strict:
                counter.update([_stem_n_gram(" ".join(gram))])
            else:
                counter.update([" ".join(gram)])

    # Convert to dataframe
    counts = pd.DataFrame.from_dict(counter, orient="index").reset_index()
    counts = counts.rename(columns={"index": "ngram", 0: "count"})

    # Add columns
    counts["session_id"] = processing_params.session_id
    counts["session_datetime"] = processing_params.session_datetime

    return counts


def _compute_ngram_usage_history(
    data: pd.DataFrame,
    ngram_size: int = 1,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Compute all ngrams usage history for the provided session dataset.
    This data can be used to plot how much of a CDP instance's discussion is comprised
    of specific keywords.

    Parameters
    ----------
    data: pd.DataFrame
        The session dataset to process and compute history for.
    ngram_size: int
        The ngram size to use for counting and calculating usage.
        Default: 1 (unigrams)
    strict: bool
        Should all ngrams be stemmed or left unstemmed for a more strict usage history.
        Default: False (stem and clean all grams in the dataset)

    Returns
    -------
    ngram_history: pd.DataFrame
        A pandas DataFrame of all found ngrams (stemmed and cleaned or unstemmed and
        uncleaned) from the data and their counts for each session and their percentage
        of use as a percent of their use for the day over the sum of all other ngrams
        used that day.
    """
    # Ensure stopwords are downloaded
    # Do this once to ensure that we don't enter a race condition
    # with multiple workers trying to download / read overtop one another
    # later on.
    try:
        from nltk.corpus import stopwords

        stopwords.words("english")
    except LookupError:
        import nltk

        nltk.download("stopwords")
        log.info("Downloaded nltk stopwords")
        from nltk.corpus import stopwords

        stopwords.words("english")

    # Construct partial for threaded counter func
    counter_func = partial(
        _count_transcript_grams,
        ngram_size=ngram_size,
        strict=strict,
    )

    # Count all uni, bi, and trigrams in transcripts
    counts = pd.concat(
        process_map(
            counter_func,
            [
                _NgramUsageProcessingParameters(
                    session_id=row.id,
                    session_datetime=row.session_datetime,
                    transcript_path=row.transcript_path,
                )
                for _, row in data.iterrows()
            ],
        )
    )

    # Make a column for just date
    counts["session_date"] = pd.to_datetime(counts["session_datetime"]).dt.date

    # Group by gram and compute combined usage for day
    counts["day_ngram_count_sum"] = counts.groupby(["ngram", "session_date"])[
        "count"
    ].transform("sum")

    # Group by date and compute total words for day
    counts["day_words_count_sum"] = counts.groupby(["session_date"])["count"].transform(
        "sum"
    )

    # Percent of word usage per day
    counts["day_ngram_percent_usage"] = (
        counts["day_ngram_count_sum"] / counts["day_words_count_sum"]
    ) * 100

    return counts


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
    compute_ngram_usage_history
        The dataset loading function which should generally paired with this function.
    """
    # Prepare ngram for user
    if not strict:
        ngram = _stem_n_gram(ngram)

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


def compute_ngram_usage_history(
    infrastructure_slug: Union[str, List[str]],
    ngram_size: int = 1,
    strict: bool = False,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Pull the minimal data needed for a session dataset for the provided infrastructure
    and start and end datetimes, then compute the ngram usage history DataFrame.

    Parameters
    ----------
    infrastructure_slug: str
        The CDP infrastructure(s) to connect to and pull sessions for.
    ngram_size: int
        The ngram size to use for counting and calculating usage.
        Default: 1 (unigrams)
    strict: bool
        Should all ngrams be stemmed or left unstemmed for a more strict usage history.
        Default: False (stem and clean all grams in the dataset)
    start_datetime: Optional[Union[str, datetime]]
        An optional datetime that the session dataset will start at.
        Default: None (no datetime beginning bound on the dataset)
    end_datetime: Optional[Union[str, datetime]]
        An optional datetime that the session dataset will end at.
        Default: None (no datetime end bound on the dataset)
    cache_dir: Optional[Union[str, Path]]
        An optional directory path to cache the dataset. Directory is created if it
        does not exist.
        Default: "./cdp-datasets"

    Returns
    -------
    gram_usage: pd.DataFrame
        A pandas DataFrame of all found ngrams (stemmed and cleaned or unstemmed and
        uncleaned) from the data and their counts for each session and their percentage
        of use as a percent of their use for the day over the sum of all other ngrams
        used that day.

    See Also
    --------
    cdp_data.datasets.get_session_dataset
        Function to pull or load a cached session dataset.
    cdp_data.keywords.plot_ngram_usage_histories
        Plot ngram usage history data.

    Notes
    -----
    This function calculates the counts and percentage of each ngram used for a day
    over the sum of all other ngrams used in that day's discussion(s). This is close but
    not exactly the same as Google's NGram Viewer: https://books.google.com/ngrams

    This function will pull a new session dataset and cache transcripts to the local
    disk in the provided (or default) cache directory.

    It is recommended to cache this dataset after computation because it may take a
    while depending on machine resources and available.
    """
    # Always cast infrastructure slugs to list for easier API
    if isinstance(infrastructure_slug, str):
        infrastructure_slug = [infrastructure_slug]

    # Create dataframe for all histories
    gram_usage = []

    # Start collecting datasets for each infrastructure
    for infra_slug in tqdm(infrastructure_slug):
        # Get the dataset
        log.info(f"Getting session dataset for {infra_slug}")
        infra_ds = datasets.get_session_dataset(
            infrastructure_slug=infra_slug,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            store_transcript=True,
            cache_dir=cache_dir,
        )

        # Compute ngram usages for infra
        log.info(f"Computing ngram history for {infra_slug}")
        infra_gram_usage = _compute_ngram_usage_history(
            infra_ds,
            ngram_size=ngram_size,
            strict=strict,
        )
        infra_gram_usage["infrastructure"] = infra_slug
        gram_usage.append(infra_gram_usage)

    # Convert gram histories to single dataframe
    return pd.concat(gram_usage)


def plot_ngram_usage_histories(
    ngram: Union[str, List[str]],
    gram_usage: pd.DataFrame,
    strict: bool = False,
    lmplot_kws: Dict[str, Any] = {},
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
    import seaborn as sns
    from matplotlib.axes import SubplotBase

    sns.set_theme(color_codes=True)

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
    for gram in tqdm(ngram):
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
    grid.add_legend()

    def _recurse_axes_grid_to_fix_datetimes(
        arr_or_subplot: Union[np.ndarray, SubplotBase],
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

    # Fix the axes to actual date formats
    _recurse_axes_grid_to_fix_datetimes(grid.axes)

    # Set axis labels
    grid.set_axis_labels("Date", "Ngram Usage (percent)")

    return grid
