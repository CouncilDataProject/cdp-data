#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import Optional, Union

import fireo
import pandas as pd
from cdp_backend.database import models as db_models
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from nltk.stem import SnowballStemmer
from cdp_backend.utils.string_utils import clean_text


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
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    # Connect to infra
    fireo.connection(client=Client(
        project=infrastructure_slug,
        credentials=AnonymousCredentials()
    ))

    # Pull ngram history
    # usage = db_models.IndexedEventGram.collection.filter(
    #     ""
    # )