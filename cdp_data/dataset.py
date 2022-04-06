#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from cdp_backend.database import models as db_models

from .utils import db_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_DATASET_STORAGE_DIR = Path(".").resolve() / "councildataproject" / "data"

###############################################################################


def get_session_dataset(
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    attach_event_metadata: bool = False,
    store_full_metadata: bool = False,
    store_transcripts: bool = False,
    store_video: bool = False,
    store_audio: bool = False,
    cache_dir: Union[str, Path] = DEFAULT_DATASET_STORAGE_DIR,
    infrastructure_slug: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get a dataset of sessions from a CDP infrastructure.
    """
    # Connect to infra
    if infrastructure_slug:
        db_utils.connect_to_database(infrastructure_slug)

    # Begin partial query
    query = db_models.Session.collection

    # Add datetime filters
    if start_datetime:
        if isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)

        query = query.filter("session_datetime", ">=", start_datetime)
    if end_datetime:
        if isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)

        query = query.filter("session_datetime", "<=", end_datetime)

    # Query for events and cast to pandas
    sessions = pd.DataFrame([e.to_dict() for e in query.fetch()])

    # Handle basic event metadata attachment
    if attach_event_metadata:
        log.info("Attaching event metadata to each session datum")
        sessions = db_utils.load_model_from_pd_columns(
            sessions,
            join_id_col="id",
            model_ref_col="event_ref",
        )

    # We only need to handle cache dir and more if any extras are True
    if not any(
        [
            store_full_metadata,
            store_transcripts,
            store_video,
            store_audio,
        ]
    ):
        return sessions

    # Handle cache dir
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

    # TODO:
    # Handle metadata reversal to ingestion model

    # Pull transcripts

    return sessions
