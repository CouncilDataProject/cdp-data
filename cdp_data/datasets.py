#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from cdp_backend.database import models as db_models
from dataclasses_json import dataclass_json
from gcsfs import GCSFileSystem
from tqdm.contrib.concurrent import thread_map

from .utils import connect_to_infrastructure, db_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_DATASET_STORAGE_DIR = Path(".").resolve() / "cdp-datasets"

###############################################################################


@dataclass
class TranscriptFetchParams:
    session_id: str
    session_key: str
    event_id: str
    transcript_selection: str
    parent_cache_dir: Path
    fs: GCSFileSystem


@dataclass_json
@dataclass
class MatchingTranscript:
    session_key: str
    transcript: db_models.Transcript
    transcript_path: Path


def _get_matching_db_transcript(
    fetch_params: TranscriptFetchParams,
) -> MatchingTranscript:
    # Get DB transcript
    db_transcript = (
        db_models.Transcript.collection.filter(
            "session_ref", "==", fetch_params.session_key
        )
        .order(f"-{fetch_params.transcript_selection}")
        .get()
    )

    # Get transcript file info
    db_transcript_file = db_transcript.file_ref.get()

    # Handle cache dir
    this_transcript_cache_dir = (
        fetch_params.parent_cache_dir
        / f"event-{fetch_params.event_id}"
        / f"session-{fetch_params.session_id}"
    )
    # Create cache dir (Handle try except because threaded)
    try:
        this_transcript_cache_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    # Download transcript if needed
    save_path = this_transcript_cache_dir / "transcript.json"
    if save_path.is_dir():
        raise IsADirectoryError(
            f"Transcript '{db_transcript_file.uri}', could not be saved because "
            f"'{save_path}' is a directory. Delete or move the directory to a "
            f"different location or change the target dataset cache dir."
        )
    elif save_path.is_file():
        log.debug(
            f"Skipping transcript '{db_transcript_file.uri}'. "
            f"A file already exists at target save path."
        )
    else:
        fetch_params.fs.get(db_transcript_file.uri, str(save_path))

    return MatchingTranscript(
        session_key=fetch_params.session_key,
        transcript=db_transcript,
        transcript_path=save_path,
    )


def get_session_dataset(
    infrastructure_slug: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    store_full_metadata: bool = False,
    store_transcript: bool = False,
    transcript_selection: str = "confidence",
    store_video: bool = False,
    store_audio: bool = False,
    cache_dir: Union[str, Path] = DEFAULT_DATASET_STORAGE_DIR,
) -> pd.DataFrame:
    """
    Get a dataset of sessions from a CDP infrastructure.
    """
    # Connect to infra
    fs = connect_to_infrastructure(infrastructure_slug)

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
            store_transcript,
            store_video,
            store_audio,
        ]
    ):
        return sessions

    # Handle cache dir
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir).resolve()

    # Make cache dir
    cache_dir = cache_dir / infrastructure_slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TODO:
    # Handle metadata reversal to ingestion model

    # Pull transcript info
    if store_transcript:
        log.info("Fetching transcripts")
        # Threaded get of transcript info
        fetched_transcript_infos = thread_map(
            _get_matching_db_transcript,
            [
                TranscriptFetchParams(
                    session_id=row.id,
                    session_key=row.key,
                    event_id=row.event.id,
                    transcript_selection=transcript_selection,
                    parent_cache_dir=cache_dir,
                    fs=fs,
                )
                for _, row in sessions.iterrows()
            ],
        )

        # Merge back to transcript dataframe
        fetched_transcripts = pd.DataFrame(
            [fti.to_dict() for fti in fetched_transcript_infos]
        )

        # Join to larger dataframe
        sessions = sessions.join(
            fetched_transcripts.set_index("session_key"),
            on="key",
        )

    return sessions
