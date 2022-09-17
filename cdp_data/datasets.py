#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from cdp_backend.database import models as db_models
from cdp_backend.utils.file_utils import resource_copy
from dataclasses_json import dataclass_json
from gcsfs import GCSFileSystem
from tqdm.contrib.concurrent import thread_map

from .constants import DEFAULT_DATASET_STORAGE_DIR
from .utils import connect_to_infrastructure, db_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Fetch utils


@dataclass
class _VideoFetchParams:
    session_id: str
    session_key: str
    event_id: str
    video_uri: str
    parent_cache_dir: Path


@dataclass_json
@dataclass
class _MatchingVideo:
    session_key: str
    video_path: Path


def _get_matching_video(
    fetch_params: _VideoFetchParams,
) -> _MatchingVideo:
    # Handle cache dir
    this_video_cache_dir = (
        fetch_params.parent_cache_dir
        / f"event-{fetch_params.event_id}"
        / f"session-{fetch_params.session_id}"
    )
    # Create cache dir (Handle try except because threaded)
    try:
        this_video_cache_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    # Download transcript if needed
    save_path = this_video_cache_dir / "video"
    if save_path.is_dir():
        raise IsADirectoryError(
            f"Video '{fetch_params.video_uri}', could not be saved because "
            f"'{save_path}' is a directory. Delete or move the directory to a "
            f"different location or change the target dataset cache dir."
        )
    elif save_path.is_file():
        log.debug(
            f"Skipping video '{fetch_params.video_uri}'. "
            f"A file already exists at target save path."
        )
    else:
        resource_copy(uri=fetch_params.video_uri, dst=save_path)

    return _MatchingVideo(
        session_key=fetch_params.session_key,
        video_path=save_path,
    )


@dataclass
class _AudioFetchParams:
    session_id: str
    session_key: str
    event_id: str
    parent_cache_dir: Path
    fs: GCSFileSystem


@dataclass_json
@dataclass
class _MatchingAudio:
    session_key: str
    audio_path: Path


def _get_matching_audio(
    fetch_params: _AudioFetchParams,
) -> _MatchingAudio:
    # Get any DB transcript
    db_transcript = db_models.Transcript.collection.filter(
        "session_ref", "==", fetch_params.session_key
    ).get()

    # Get transcript file info
    db_transcript_file = db_transcript.file_ref.get()

    # Strip the transcript details from filename
    # Audio files are stored with the same URI as the transcript
    # but instead of `-cdp_{version}-transcript.json`
    # they simply end with `-audio.wav`
    transcript_uri_parts = db_transcript_file.uri.split("/")
    transcript_filename = transcript_uri_parts[-1]
    uri_base = "/".join(transcript_uri_parts[:-1])
    session_content_hash = transcript_filename[:64]
    audio_uri = "/".join([uri_base, f"{session_content_hash}-audio.wav"])

    # Handle cache dir
    this_audio_cache_dir = (
        fetch_params.parent_cache_dir
        / f"event-{fetch_params.event_id}"
        / f"session-{fetch_params.session_id}"
    )
    # Create cache dir (Handle try except because threaded)
    try:
        this_audio_cache_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    # Download audio if needed
    save_path = this_audio_cache_dir / "audio.wav"
    if save_path.is_dir():
        raise IsADirectoryError(
            f"Audio '{audio_uri}', could not be saved because "
            f"'{save_path}' is a directory. Delete or move the directory to a "
            f"different location or change the target dataset cache dir."
        )
    elif save_path.is_file():
        log.debug(
            f"Skipping audio '{audio_uri}'. "
            f"A file already exists at target save path."
        )
    else:
        fetch_params.fs.get(audio_uri, str(save_path))

    return _MatchingAudio(
        session_key=fetch_params.session_key,
        audio_path=save_path,
    )


@dataclass
class _TranscriptFetchParams:
    session_id: str
    session_key: str
    event_id: str
    transcript_selection: str
    parent_cache_dir: Path
    fs: GCSFileSystem


@dataclass_json
@dataclass
class _MatchingTranscript:
    session_key: str
    transcript: db_models.Transcript
    transcript_path: Path


def _get_matching_db_transcript(
    fetch_params: _TranscriptFetchParams,
) -> _MatchingTranscript:
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

    return _MatchingTranscript(
        session_key=fetch_params.session_key,
        transcript=db_transcript,
        transcript_path=save_path,
    )


def _merge_dataclasses_to_df(
    data_objs: List[dataclass_json],
    df: pd.DataFrame,
    data_objs_key: str,
    df_key: str,
) -> pd.DataFrame:
    # Merge back to video dataframe
    fetched_objs = pd.DataFrame([obj.to_dict() for obj in data_objs])

    # Join to larger dataframe
    return df.join(
        fetched_objs.set_index(data_objs_key),
        on=df_key,
    )


###############################################################################


def get_session_dataset(
    infrastructure_slug: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    store_full_metadata: bool = False,
    store_transcript: bool = False,
    transcript_selection: str = "confidence",
    store_video: bool = False,
    store_audio: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Get a dataset of sessions from a CDP infrastructure.

    Parameters
    ----------
    infrastructure_slug: str
        The CDP infrastructure to connect to and pull sessions for.
    start_datetime: Optional[Union[str, datetime]]
        An optional datetime that the session dataset will start at.
        Default: None (no datetime beginning bound on the dataset)
    end_datetime: Optional[Union[str, datetime]]
        An optional datetime that the session dataset will end at.
        Default: None (no datetime end bound on the dataset)
    store_full_metadata: bool
        Should a JSON file of the full event metadata be stored to disk and a
        path to the stored JSON file be added to the returned DataFrame.
        Default: False (do not request extra data and store to disk)
        **Currently not implemented**
    store_transcript: bool
        Should a session transcript be requested and stored to disk and a path
        to the stored transcript JSON file be added to the returned DataFrame.
        Default: False (do not request extra data and do not store the transcript)
    transcript_selection: str
        How should the single transcript be selected.
        Options: "confidence" for transcript with the highest confidence value,
        "created" for the most recently created transcript.
        Default: "confidence" (Return the single highest confidence
        transcript per session)
    store_video: bool
        Should the session video be requested and stored to disk and a path to the
        stored video file be added to the returned DataFrame. Note: the video is stored
        without a file extension. However, the video with always be either mp4 or webm.
        Default: False (do not request and store the video)
    store_audio: bool
        Should the session audio be requested and stored to disk and a path to the
        stored audio file be added to the returned DataFrame.
        Default: False (do not request and store the audio)
    cache_dir: Optional[Union[str, Path]]
        An optional directory path to cache the dataset. Directory is created if it
        does not exist.
        Default: "./cdp-datasets"

    Returns
    -------
    dataset: pd.DataFrame
        The dataset with all additions requested.

    Notes
    -----
    All file additions (transcript, full event metadata, video, audio, etc.) are cached
    to disk to avoid multiple downloads. If you use the same cache directory multiple
    times over the course of multiple runs, no new data will be downloaded, but the
    existing files will be used. Caching is done by simply file existence not by a
    content hash comparison.

    Datasets are cached with the following structure::

        {cache-dir}/
        └── {infrastructure_slug}
            ├── event-{event-id-0}
            │   ├── metadata.json
            │   └── session-{session-id-0}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video
            ├── event-{event-id-1}
            │   ├── metadata.json
            │   └── session-{session-id-0}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video
            ├── event-{event-id-2}
            │   ├── metadata.json
            │   └── session-{session-id-0}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video
            │   └── session-{session-id-1}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video

    To clean a whole dataset or specific events or sessions simply delete the
    associated directory.
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

    # If no sessions found, return empty dataset
    if len(sessions) == 0:
        return pd.DataFrame(
            columns=[
                "session_datetime",
                "session_index",
                "session_content_hash",
                "video_uri",
                "caption_uri",
                "external_source_id",
                "id",
                "key",
                "event",
            ]
        )

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
    if not cache_dir:
        cache_dir = DEFAULT_DATASET_STORAGE_DIR
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir).resolve()

    # Make cache dir
    cache_dir = cache_dir / infrastructure_slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TODO:
    # Handle metadata reversal to ingestion model
    if store_full_metadata:
        log.warning("`store_full_metadata` not implemented")

    # Handle video
    if store_video:
        log.info("Fetching video")
        fetched_video_infos = thread_map(
            _get_matching_video,
            [
                _VideoFetchParams(
                    session_id=row.id,
                    session_key=row.key,
                    event_id=row.event.id,
                    video_uri=row.video_uri,
                    parent_cache_dir=cache_dir,
                )
                for _, row in sessions.iterrows()
            ],
            desc="Fetching videos",
        )

        # Merge fetched data back to session df
        sessions = _merge_dataclasses_to_df(
            data_objs=fetched_video_infos,
            df=sessions,
            data_objs_key="session_key",
            df_key="key",
        )

    # Handle audio
    if store_audio:
        log.info("Fetching audio")
        fetched_audio_infos = thread_map(
            _get_matching_audio,
            [
                _AudioFetchParams(
                    session_id=row.id,
                    session_key=row.key,
                    event_id=row.event.id,
                    parent_cache_dir=cache_dir,
                    fs=fs,
                )
                for _, row in sessions.iterrows()
            ],
            desc="Fetching audios",
        )

        # Merge fetched data back to session df
        sessions = _merge_dataclasses_to_df(
            data_objs=fetched_audio_infos,
            df=sessions,
            data_objs_key="session_key",
            df_key="key",
        )

    # Pull transcript info
    if store_transcript:
        log.info("Fetching transcripts")
        # Threaded get of transcript info
        fetched_transcript_infos = thread_map(
            _get_matching_db_transcript,
            [
                _TranscriptFetchParams(
                    session_id=row.id,
                    session_key=row.key,
                    event_id=row.event.id,
                    transcript_selection=transcript_selection,
                    parent_cache_dir=cache_dir,
                    fs=fs,
                )
                for _, row in sessions.iterrows()
            ],
            desc="Fetching transcripts",
        )

        # Merge fetched data back to session df
        sessions = _merge_dataclasses_to_df(
            data_objs=fetched_transcript_infos,
            df=sessions,
            data_objs_key="session_key",
            df_key="key",
        )

    return sessions
