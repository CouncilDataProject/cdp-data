#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from cdp_backend.database import models as db_models
from cdp_backend.pipeline.transcript_model import Transcript
from cdp_backend.utils.string_utils import clean_text
from nltk import ngrams
from nltk.stem import SnowballStemmer
from tqdm.contrib.concurrent import process_map

from .utils import db_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class NgramUsageProcessingParameters:
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


# TODO:
# Write function to compute full ngram history not "relevancy over time"
#
# From Google Ngram Viewer:
# https://books.google.com/ngrams/info
# This shows trends in three ngrams from 1960 to 2015: "nursery school"
# (a 2-gram or bigram), "kindergarten" (a 1-gram or unigram), and
# "child care" (another bigram). What the y-axis shows is this: of all the bigrams
# contained in our sample of books written in English and published in the
# United States, what percentage of them are "nursery school" or "child care"?
# Of all the unigrams, what percentage of them are "kindergarten"? Here, you can
# see that use of the phrase "child care" started to rise in the late 1960s,
# overtaking "nursery school" around 1970 and then "kindergarten" around 1973.
# It peaked shortly after 1990 and has been falling steadily since.
#
# TF-IDF shows the deviation from normal whereas percent of total shows
# trends in total discussion of Ngram


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

    Notes
    -----
    This function pulls the TF-IDF (or other future indexed values) score for the
    provided ngram over time. This is a measure of relevancy to a document and not
    the same as Google's NGram Viewer which shows what percentage of literature
    used the term.

    See Also
    --------
    cdp_data.keywords.compute_ngram_usage_history
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

    # TODO: add data agg by num days

    return ngram_history


def _prepare_ngram_history_plotting_data(
    data: pd.DataFrame,
    ngram_col: str,
    value_col: str,
    dt_col: str,
) -> pd.DataFrame:
    # Select down to just the columns we want
    # Reset index
    # Sort values by datetime
    subset = (
        data[[ngram_col, value_col, dt_col]]
        .sort_values([ngram_col, dt_col])
        .reset_index(drop=True)
    )

    # First group data by ngram and date and use max for the day
    subset[dt_col] = pd.to_datetime(subset[dt_col])
    subset = (
        subset.groupby([ngram_col, pd.Grouper(key=dt_col, freq="D")])
        .max()
        .reset_index()
    ).replace([None])

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
        subset.set_index(dt_col)
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
    Prepare an ngram history DataFrame specifically for plotting.
    This function will subset the DataFrame to just the provided columns,
    will only store a single value for each day if there are multiple
    (keeping the max value), and finally filling all missing days with zero values.

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
        The subset and missing dates filled DataFrame.

    See Also
    --------
    get_ngram_relevancy_history
    """
    return _prepare_ngram_history_plotting_data(
        data=data,
        ngram_col=ngram_col,
        value_col=value_col,
        dt_col=dt_col,
    )


def _count_transcript_grams(
    processing_params: NgramUsageProcessingParameters,
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


def compute_ngram_usage_history(
    data: pd.DataFrame,
    ngram_size: int = 1,
    strict: bool = False,
) -> pd.DataFrame:
    """
    From a session dataset with transcripts available, compute an ngram's usage history.
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

    # TODO:
    # maybe daskify?

    # Count all uni, bi, and trigrams in transcripts
    counts = pd.concat(
        process_map(
            counter_func,
            [
                NgramUsageProcessingParameters(
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
    )

    return counts


def prepare_ngram_usage_history_plotting_data(
    ngram: str,
    data: pd.DataFrame,
    strict: bool = False,
    ngram_col: str = "ngram",
    percent_col: str = "day_ngram_percent_usage",
    dt_col: str = "session_date",
) -> pd.DataFrame:
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
    )
