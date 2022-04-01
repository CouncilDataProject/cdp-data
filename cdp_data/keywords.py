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


def get_ngram_usage(
    infrastructure_slug: str,
    ngram: str,
    strict: bool = False,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
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
        from tqdm import tqdm

        # Register `pandas.progress_apply` and `pandas.Series.map_apply`
        tqdm.pandas(desc="Event data attached to ngram usage")
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
