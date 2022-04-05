#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fireo
import pandas as pd
from cdp_backend.database import models as db_models
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from tqdm.contrib.concurrent import thread_map

from .utils import db_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_DATASET_STORAGE_DIR = Path(".").resolve() / "councildataproject" / "data"

###############################################################################


def get_session_dataset(
    infrastructure_slug: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    attach_event_metadata: bool = False,
    store_full_metadata: bool = False,
    store_transcripts: bool = False,
    store_video: bool = False,
    store_audio: bool = False,
    cache_dir: Union[str, Path] = DEFAULT_DATASET_STORAGE_DIR,
) -> pd.DataFrame:
    """
    Get a dataset of sessions from a CDP infrastructure.
    """
    # Connect to infra
    fireo.connection(
        client=Client(project=infrastructure_slug, credentials=AnonymousCredentials())
    )

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
        sessions["event"] = thread_map(
            db_utils.load_from_model_reference,
            sessions.event_ref,
        )
        sessions = sessions.drop(["event_ref"], axis=1)

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
