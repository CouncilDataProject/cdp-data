#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from typing import Optional, Union

import fireo
import pandas as pd
from cdp_backend.database import models as db_models
from cdp_backend.utils.string_utils import clean_text
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from nltk.stem import SnowballStemmer

###############################################################################

log = logging.getLogger(__name__)

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
    infrastructure_slug: str,
    strict: bool = False,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """
    Pull an n-gram's relevancy history from a CDP database.

    Parameters
    ----------
    ngram: str
        The unigram, bigram, or trigram to retrieve history for.
    infrastructure_slug: str
        The CDP infrastructure slug to connect to.
    strict: bool
        Should the provided ngram be used for a strict "unstemmed_gram" query or not.
        Default: False (stem and clean the ngram before querying)
    start_datetime: Optional[Union[str, datetime]]
        The earliest possible datetime for ngram history to be retrieved for.
        If provided as a string, the datetime should be in ISO format.
    end_datetime: Optional[Union[str, datetime]]
        The latest possible datetime for ngram history to be retrieved for.
        If provided as a string, the datetime should be in ISO format.

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
    fireo.connection(
        client=Client(project=infrastructure_slug, credentials=AnonymousCredentials())
    )

    # Begin partial query
    keywords_collection = db_models.IndexedEventGram.collection

    # TODO:
    # Add datetime filtering

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
    try:
        # Register `pandas.progress_apply` and `pandas.Series.map_apply`
        from tqdm import tqdm

        tqdm.pandas(desc="Event attachment")

        ngram_history["event"] = ngram_history.progress_apply(
            lambda row: row.event_ref.get(),
            axis=1,
        )
    except ImportError:
        ngram_history["event"] = ngram_history.apply(
            lambda row: row.event_ref.get(),
            axis=1,
        )

    # Unpack event cols
    ngram_history["event_datetime"] = ngram_history.apply(
        lambda row: row.event.event_datetime,
        axis=1,
    )

    # Drop non-needed cols
    ngram_history = ngram_history.drop(["event_ref"], axis=1)

    return ngram_history

def compute_ngram_usage_history(
    ngram: str,
    infrastructure_slug: str,
    strict: bool = False,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """
    Pull transcript and event data for the provided infrastructure
    and compute ngram usage over time.
    """
    pass