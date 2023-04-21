#!/usr/bin/env python

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from cdp_backend.database import models as db_models
from cdp_backend.pipeline.transcript_model import Transcript
from cdp_backend.utils.file_utils import resource_copy
from dataclasses_json import dataclass_json
from fireo.models import Model
from gcsfs import GCSFileSystem
from tqdm.contrib.concurrent import process_map, thread_map

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
    fs: GCSFileSystem
    raise_on_error: bool


@dataclass_json
@dataclass
class _MatchingVideo:
    session_key: str
    video_path: Optional[Path]


def _get_matching_video(
    fetch_params: _VideoFetchParams,
) -> _MatchingVideo:
    try:
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

    except Exception as e:
        if fetch_params.raise_on_error:
            raise FileNotFoundError(
                f"Something went wrong while fetching the video for session: "
                f"'{fetch_params.session_id}' from '{fetch_params.fs.project}' "
                f"Please check if a report has already been made to GitHub "
                f"(https://github.com/CouncilDataProject/cdp-data/issues). "
                f"If you cannot find an open issue for this session, "
                f"please create a new one. "
                f"In the meantime, please try rerunning your request with "
                f"`raise_on_error=False`"
            ) from e

        return _MatchingVideo(
            session_key=fetch_params.session_key,
            video_path=None,
        )


@dataclass
class _AudioFetchParams:
    session_id: str
    session_key: str
    event_id: str
    parent_cache_dir: Path
    fs: GCSFileSystem
    raise_on_error: bool


@dataclass_json
@dataclass
class _MatchingAudio:
    session_key: str
    audio_path: Optional[Path]


def _get_matching_audio(
    fetch_params: _AudioFetchParams,
) -> _MatchingAudio:
    try:
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

    except Exception as e:
        if fetch_params.raise_on_error:
            raise FileNotFoundError(
                f"Something went wrong while fetching the video for session: "
                f"'{fetch_params.session_id}' from '{fetch_params.fs.project}' "
                f"Please check if a report has already been made to GitHub "
                f"(https://github.com/CouncilDataProject/cdp-data/issues). "
                f"If you cannot find an open issue for this session, "
                f"please create a new one. "
                f"In the meantime, please try rerunning your request with "
                f"`raise_on_error=False`"
            ) from e

        return _MatchingAudio(
            session_key=fetch_params.session_key,
            audio_path=None,
        )


@dataclass
class _TranscriptFetchParams:
    session_id: str
    session_key: str
    event_id: str
    transcript_selection: str
    parent_cache_dir: Path
    fs: GCSFileSystem
    raise_on_error: bool


@dataclass_json
@dataclass
class _MatchingTranscript:
    session_key: str
    transcript: Optional[db_models.Transcript]
    transcript_path: Optional[Path]


def _get_matching_db_transcript(
    fetch_params: _TranscriptFetchParams,
) -> _MatchingTranscript:
    try:
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

    except Exception as e:
        if fetch_params.raise_on_error:
            raise FileNotFoundError(
                f"Something went wrong while fetching the transcript for session: "
                f"'{fetch_params.session_id}' from '{fetch_params.fs.project}' "
                f"Please check if a report has already been made to GitHub "
                f"(https://github.com/CouncilDataProject/cdp-data/issues). "
                f"If you cannot find an open issue for this session, "
                f"please create a new one. "
                f"In the meantime, please try rerunning your request with "
                f"`raise_on_error=False`"
            ) from e

        return _MatchingTranscript(
            session_key=fetch_params.session_key,
            transcript=None,
            transcript_path=None,
        )


@dataclass_json
@dataclass
class _TranscriptConversionParams:
    session_id: str
    session_key: str
    fs: GCSFileSystem
    transcript_path: Path
    raise_on_error: bool


@dataclass_json
@dataclass
class _ConvertedTranscript:
    session_key: str
    transcript_as_csv_path: Optional[Path]


def _convert_transcript_to_csv(
    convert_params: _TranscriptConversionParams,
) -> _ConvertedTranscript:
    # Safety around conversion, any error, we return None and that will be dropped
    try:
        # Get storage name
        dest = convert_params.transcript_path.with_suffix(".csv")

        # If this path already exists, just attach and return
        if dest.exists():
            return _ConvertedTranscript(
                session_key=convert_params.session_key,
                transcript_as_csv_path=dest,
            )

        # The path doesn't exist, convert
        transcript_df = convert_transcript_to_dataframe(convert_params.transcript_path)
        transcript_df.to_csv(dest, index=False)

        return _ConvertedTranscript(
            session_key=convert_params.session_key,
            transcript_as_csv_path=dest,
        )

    except Exception as e:
        if convert_params.raise_on_error:
            raise ValueError(
                f"Something went wrong while converting the transcript for a session: "
                f"'{convert_params.session_id}' from '{convert_params.fs.project}' "
                f"Please check if a report has already been made to GitHub "
                f"(https://github.com/CouncilDataProject/cdp-data/issues). "
                f"If you cannot find an open issue for this session, "
                f"please create a new one. "
                f"In the meantime, please try rerunning your request with "
                f"`raise_on_error=False`"
            ) from e

        return _ConvertedTranscript(
            session_key=convert_params.session_key,
            transcript_as_csv_path=None,
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


def replace_db_model_cols_with_id_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace all database model column values with the model ID.

    Example: an `event` column with event models, will be replaced by an
    `event_id` column with just the event id.

    Parameters
    ----------
    df: pd.DataFrame
        The data to replace database models with just ids.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Get a single row sample
    sample = df.loc[0]

    # Iter over cols, check type, and replace if type is a db model
    for col in df.columns:
        sample_col_val = sample[col]
        if isinstance(sample_col_val, Model):
            df[f"{sample_col_val.collection_name}_id"] = df[col].apply(lambda m: m.id)
            df = df.drop(columns=[col])

    return df


def replace_pathlib_path_cols_with_str_path_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace all pathlib Path column values with string column values.

    Example: a `transcript_path` column with a pathlib Path, will be replaced
    as a normal Python string.

    Parameters
    ----------
    df: pd.DataFrame
        The data to replace paths in.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.
    """
    # Get a single row sample
    sample = df.loc[0]

    # Iter over cols, check type, and replace if type is a db model
    for col in df.columns:
        sample_col_val = sample[col]
        if isinstance(sample_col_val, Path):
            df[col] = df[col].apply(lambda p: str(p))

    return df


def replace_dataframe_cols_with_storage_replacements(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run various replacement functions against the dataframe to get the
    data to a point where it can be store to disk.

    Parameters
    ----------
    df: pd.DataFrame
        The data to fix.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame.

    See Also
    --------
    replace_db_model_cols_with_id_cols
        Function to replace database model column values with their ids.
    replace_pathlib_path_cols_with_str_path_cols
        Function to replace pathlib Path column values with normal Python strings.
    """
    # Replace everything
    for func in (
        replace_db_model_cols_with_id_cols,
        replace_pathlib_path_cols_with_str_path_cols,
    ):
        df = func(df)

    return df


def convert_transcript_to_dataframe(
    transcript: Union[str, Path, Transcript]
) -> pd.DataFrame:
    """
    Create a dataframe from only the sentence data from the provided transcript.

    Parameters
    ----------
    transcript: Union[str, Path, Transcript]
        The transcript to pull all sentences from.

    Returns
    -------
    pd.DataFrame:
        The sentences of the transcript.
    """
    # Read transcript is need be
    if isinstance(transcript, (str, Path)):
        with open(transcript) as open_f:
            transcript = Transcript.from_json(open_f.read())

    # Dump sentences to frame
    sentences = pd.DataFrame(transcript.sentences)

    # Drop the words col
    sentences = sentences.drop(columns=["words"])
    return sentences


def get_session_dataset(  # noqa: C901
    infrastructure_slug: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    replace_py_objects: bool = False,
    store_full_metadata: bool = False,
    store_transcript: bool = False,
    transcript_selection: str = "created",
    store_transcript_as_csv: bool = False,
    store_video: bool = False,
    store_audio: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    raise_on_error: bool = True,
    tqdm_kws: Union[Dict[str, Any], None] = None,
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
    replace_py_objects: bool
        Replace any non-standard Python type with standard ones to
        allow the returned data be ready for storage.
        See 'See Also' for more details.
        Default: False (keep Python objects in the DataFrame)
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
        Default: "created" (Return the most recently created
        transcript per session)
    store_transcript_as_csv: bool
        Additionally convert and store all transcripts as CSVs.
        Does nothing if `store_transcript` is False.
        Default: False (do not convert and store again)
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
    raise_on_error: bool
        Should any failure to pull files result in an error or be ignored.
        Default: True (raise on any failure)
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

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
            │   ├── metadata.json
            │   └── session-{session-id-0}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video
            ├── event-{event-id-1}
            │   ├── metadata.json
            │   └── session-{session-id-0}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video
            ├── event-{event-id-2}
            │   ├── metadata.json
            │   └── session-{session-id-0}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video
            │   └── session-{session-id-1}
            │       ├── audio.wav
            │       ├── transcript.json
            │       └── video

    To clean a whole dataset or specific events or sessions simply delete the
    associated directory.

    See Also
    --------
    replace_dataframe_cols_with_storage_replacements
        The function used to clean the data of non-standard Python types.
    """
    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

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
        tqdm_kws=tqdm_kws,
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
                    fs=fs,
                    raise_on_error=raise_on_error,
                )
                for _, row in sessions.iterrows()
            ],
            desc="Fetching videos",
            **tqdm_kws,
        )

        # Merge fetched data back to session df
        sessions = _merge_dataclasses_to_df(
            data_objs=fetched_video_infos,
            df=sessions,
            data_objs_key="session_key",
            df_key="key",
        ).dropna(subset=["video_path"])

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
                    raise_on_error=raise_on_error,
                )
                for _, row in sessions.iterrows()
            ],
            desc="Fetching audios",
            **tqdm_kws,
        )

        # Merge fetched data back to session df
        sessions = _merge_dataclasses_to_df(
            data_objs=fetched_audio_infos,
            df=sessions,
            data_objs_key="session_key",
            df_key="key",
        ).dropna(subset=["audio_path"])

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
                    raise_on_error=raise_on_error,
                )
                for _, row in sessions.iterrows()
            ],
            desc="Fetching transcripts",
            **tqdm_kws,
        )

        # Merge fetched data back to session df
        sessions = _merge_dataclasses_to_df(
            data_objs=fetched_transcript_infos,
            df=sessions,
            data_objs_key="session_key",
            df_key="key",
        ).dropna(subset=["transcript_path"])

        # Handle conversion of transcripts to CSVs
        if store_transcript_as_csv:
            log.info("Converting and storing transcripts as CSVs")

            # Threaded processing of transcript conversion
            converted_transcript_infos = process_map(
                _convert_transcript_to_csv,
                [
                    _TranscriptConversionParams(
                        session_id=row.id,
                        session_key=row.key,
                        fs=fs,
                        transcript_path=row.transcript_path,
                        raise_on_error=raise_on_error,
                    )
                    for _, row in sessions.iterrows()
                ],
                desc="Converting transcripts",
                **tqdm_kws,
            )

            # Merge fetched data back to session df
            sessions = _merge_dataclasses_to_df(
                data_objs=converted_transcript_infos,
                df=sessions,
                data_objs_key="session_key",
                df_key="key",
            ).dropna(subset=["transcript_as_csv_path"])

    # Replace col values with storage ready replacements
    if replace_py_objects:
        sessions = replace_dataframe_cols_with_storage_replacements(sessions)

    return sessions


def save_dataset(
    df: pd.DataFrame,
    dest: Union[str, Path],
) -> Path:
    """
    Helper function to store a dataset to disk by replacing
    non-standard Python types with storage ready replacements.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to store.
    dest: Union[str, Path]
        The path to store the data. Must end in ".csv" or ".parquet".

    Returns
    -------
    Path:
        The path to the stored data.

    See Also
    --------
    replace_dataframe_cols_with_storage_replacements
        The function used to replace column values.
    """
    # Replace col values with storage ready replacements
    df = replace_dataframe_cols_with_storage_replacements(df)

    # Convert dest to Path
    if isinstance(dest, str):
        dest = Path(dest)

    # Check suffix
    if dest.suffix == ".csv":
        df.to_csv(dest, index=False)
        return dest

    if dest.suffix == ".parquet":
        df.to_parquet(dest)
        return dest

    raise ValueError(
        f"Unrecognized filepath suffix: '{dest.suffix}'. "
        f"Support storage types are 'csv' and 'parquet'."
    )


def _get_votes_for_event(key: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            v.to_dict()
            for v in db_models.Vote.collection.filter(
                "event_ref",
                "==",
                key,
            ).fetch()
        ]
    )


def get_vote_dataset(
    infrastructure_slug: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    replace_py_objects: bool = False,
    tqdm_kws: Union[Dict[str, Any], None] = None,
) -> pd.DataFrame:
    """
    Get a dataset of votes from a CDP infrastructure.

    Parameters
    ----------
    infrastructure_slug: str
        The CDP infrastructure to connect to and pull votes for.
    start_datetime: Optional[Union[str, datetime]]
        An optional datetime that the vote dataset will start at.
        Default: None (no datetime beginning bound on the dataset)
    end_datetime: Optional[Union[str, datetime]]
        An optional datetime that the vote dataset will end at.
        Default: None (no datetime end bound on the dataset)
    replace_py_objects: bool
        Replace any non-standard Python type with standard ones to
        allow the returned data be ready for storage.
        See 'See Also' for more details.
        Default: False (keep Python objects in the DataFrame)
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

    Returns
    -------
    pd.DataFrame
        The dataset requested.

    See Also
    --------
    replace_dataframe_cols_with_storage_replacements
        The function used to clean the data of non-standard Python types.
    """
    if not tqdm_kws:
        tqdm_kws = {}

    # Connect to infra
    connect_to_infrastructure(infrastructure_slug)

    # Begin partial query
    query = db_models.Event.collection

    # Add datetime filters
    if start_datetime:
        if isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)

        query = query.filter("event_datetime", ">=", start_datetime)
    if end_datetime:
        if isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)

        query = query.filter("event_datetime", "<=", end_datetime)

    # Query for events
    events = list(query.fetch())

    # Thread fetch votes for each event
    log.info("Fetching votes for each event")
    fetched_votes_frames = thread_map(
        _get_votes_for_event,
        [e.key for e in events],
        desc="Fetching votes for each event",
        **tqdm_kws,
    )
    votes = pd.concat(fetched_votes_frames)

    # If no votes are found, return empty dataset
    if len(votes) == 0:
        return pd.DataFrame(
            columns=[
                "decision",
                "in_majority",
                "external_source_id",
                "id",
                "key",
                "event_id",
                "event_key",
                "event_datetime",
                "agenda_uri",
                "minutes_uri",
                "matter_id",
                "matter_key",
                "matter_name",
                "matter_type",
                "matter_title",
                "event_minutes_item_id",
                "event_minutes_item_key",
                "event_minutes_item_index_in_meeting",
                "event_minutes_item_overall_decision",
                "person_id",
                "person_key",
                "person_name",
                "body_id",
                "body_key",
                "body_name",
            ],
        )

    # Thread fetch events for each vote
    log.info("Attaching event metadata to each vote datum")
    votes = db_utils.load_model_from_pd_columns(
        votes,
        join_id_col="id",
        model_ref_col="event_ref",
        tqdm_kws=tqdm_kws,
    )

    # Thread fetch matters for each vote
    log.info("Attaching matter metadata to each vote datum")
    votes = db_utils.load_model_from_pd_columns(
        votes,
        join_id_col="id",
        model_ref_col="matter_ref",
        tqdm_kws=tqdm_kws,
    )

    # Thread fetch event minutes items for each vote
    log.info("Attaching event minutes item metadata to each vote datum")
    votes = db_utils.load_model_from_pd_columns(
        votes,
        join_id_col="id",
        model_ref_col="event_minutes_item_ref",
        tqdm_kws=tqdm_kws,
    )

    # Thread fetch people for each vote
    log.info("Attaching person metadata to each vote datum")
    votes = db_utils.load_model_from_pd_columns(
        votes,
        join_id_col="id",
        model_ref_col="person_ref",
        tqdm_kws=tqdm_kws,
    )

    # Expand event models
    votes = db_utils.expand_models_from_pd_column(
        votes,
        model_col="event",
        model_attr_rename_lut={
            "id": "event_id",
            "key": "event_key",
            "body_ref": "body_ref",
            "event_datetime": "event_datetime",
            "agenda_uri": "agenda_uri",
            "minutes_uri": "minutes_uri",
        },
        tqdm_kws=tqdm_kws,
    )

    # Expand matter models
    votes = db_utils.expand_models_from_pd_column(
        votes,
        model_col="matter",
        model_attr_rename_lut={
            "id": "matter_id",
            "key": "matter_key",
            "name": "matter_name",
            "matter_type": "matter_type",
            "title": "matter_title",
        },
        tqdm_kws=tqdm_kws,
    )

    # Expand event minutes item models
    votes = db_utils.expand_models_from_pd_column(
        votes,
        model_col="event_minutes_item",
        model_attr_rename_lut={
            "id": "event_minutes_item_id",
            "key": "event_minutes_item_key",
            "index": "event_minutes_item_index_in_meeting",
            "decision": "event_minutes_item_overall_decision",
        },
        tqdm_kws=tqdm_kws,
    )

    # Expand person models
    votes = db_utils.expand_models_from_pd_column(
        votes,
        model_col="person",
        model_attr_rename_lut={
            "id": "person_id",
            "key": "person_key",
            "name": "person_name",
        },
        tqdm_kws=tqdm_kws,
    )

    # Thread fetch body for each vote
    log.info("Attaching body metadata to each vote datum")
    votes = db_utils.load_model_from_pd_columns(
        votes,
        join_id_col="id",
        model_ref_col="body_ref",
        tqdm_kws=tqdm_kws,
    )

    # Expand body models
    votes = db_utils.expand_models_from_pd_column(
        votes,
        model_col="body",
        model_attr_rename_lut={
            "id": "body_id",
            "key": "body_key",
            "name": "body_name",
        },
        tqdm_kws=tqdm_kws,
    )

    # Replace col values with storage ready replacements
    if replace_py_objects:
        votes = replace_dataframe_cols_with_storage_replacements(votes)

    return votes
