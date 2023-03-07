#!/usr/bin/env python

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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
from .utils.incremental_average import IncrementalStats

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from torch import Tensor

###############################################################################

# Logging
log = logging.getLogger(__name__)

###############################################################################
# Constants / Support Classes


@dataclass
class _TranscriptProcessingParams:
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
    cdp_data.plotting.prepare_ngram_history_plotting_data
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


def _count_transcript_grams(
    processing_params: _TranscriptProcessingParams,
    ngram_size: int,
    strict: bool,
) -> pd.DataFrame:
    # Load transcript
    with open(processing_params.transcript_path) as open_f:
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
    tqdm_kws: Union[Dict[str, Any], None] = None,
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
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

    Returns
    -------
    ngram_history: pd.DataFrame
        A pandas DataFrame of all found ngrams (stemmed and cleaned or unstemmed and
        uncleaned) from the data and their counts for each session and their percentage
        of use as a percent of their use for the day over the sum of all other ngrams
        used that day.
    """
    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

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
                _TranscriptProcessingParams(
                    session_id=row.id,
                    session_datetime=row.session_datetime,
                    transcript_path=row.transcript_path,
                )
                for _, row in data.iterrows()
            ],
            desc="Counting ngrams in each transcript",
            **tqdm_kws,
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


def compute_ngram_usage_history(
    infrastructure_slug: Union[str, List[str]],
    ngram_size: int = 1,
    strict: bool = False,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    raise_on_error: bool = True,
    tqdm_kws: Union[Dict[str, Any], None] = None,
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
    raise_on_error: bool
        Should any failure to pull files result in an error or be ignored.
        Default: True (raise on any failure)
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

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
    cdp_data.plotting.plot_ngram_usage_histories
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
    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

    # Always cast infrastructure slugs to list for easier API
    if isinstance(infrastructure_slug, str):
        infrastructure_slug = [infrastructure_slug]

    # Create dataframe for all histories
    gram_usage = []

    # Start collecting datasets for each infrastructure
    for infra_slug in tqdm(
        infrastructure_slug,
        desc="Counting ngrams for each infrastructure",
        **tqdm_kws,
    ):
        # Get the dataset
        log.info(f"Getting session dataset for {infra_slug}")
        infra_ds = datasets.get_session_dataset(
            infrastructure_slug=infra_slug,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            store_transcript=True,
            cache_dir=cache_dir,
            raise_on_error=raise_on_error,
            tqdm_kws=tqdm_kws,
        )

        # Compute ngram usages for infra
        log.info(f"Computing ngram history for {infra_slug}")
        infra_gram_usage = _compute_ngram_usage_history(
            infra_ds,
            ngram_size=ngram_size,
            strict=strict,
            tqdm_kws=tqdm_kws,
        )
        infra_gram_usage["infrastructure"] = infra_slug
        gram_usage.append(infra_gram_usage)

    # Convert gram histories to single dataframe
    return pd.concat(gram_usage)


def _compute_transcript_sim_stats(
    processing_params: _TranscriptProcessingParams,
    query_vec: "Tensor",
    model: "SentenceTransformer",
) -> pd.DataFrame:
    from sentence_transformers.util import cos_sim

    # Load transcript
    with open(processing_params.transcript_path) as open_f:
        transcript = Transcript.from_json(open_f.read())

    # Create incremental stats for updating mean, min, max
    inc_stats = IncrementalStats()

    # For each sentence in transcript,
    # Embed and calc similarity
    # Track min, max, and update mean
    for sentence in transcript.sentences:
        sentence_enc = model.encode(sentence.text)
        query_sim = cos_sim(
            query_vec, sentence_enc
        ).item()  # cos_sim returns a 2d tensor
        inc_stats.add(query_sim)

    # Convert to dataframe
    return pd.DataFrame(
        [
            {
                "session_id": processing_params.session_id,
                "session_datetime": processing_params.session_datetime,
                "similarity_min": inc_stats.current_min,
                "similarity_max": inc_stats.current_max,
                "similarity_mean": inc_stats.current_mean,
            }
        ]
    )


def _compute_query_semantic_similarity_history(
    data: pd.DataFrame,
    query_vec: "Tensor",
    model: "SentenceTransformer",
    tqdm_kws: Union[Dict[str, Any], None] = None,
) -> pd.DataFrame:
    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

    # Construct partial for threaded counter func
    process_func = partial(
        _compute_transcript_sim_stats,
        query_vec=query_vec,
        model=model,
    )

    # Compute min, max, and mean cos_sim using the sentences
    # of each transcript
    sim_stats_list: List[pd.DataFrame] = []
    for _, row in tqdm(
        data.iterrows(),
        desc="Computing semantic similarity for each transcript",
        **tqdm_kws,
    ):
        sim_stats_list.append(
            process_func(
                _TranscriptProcessingParams(
                    session_id=row.id,
                    session_datetime=row.session_datetime,
                    transcript_path=row.transcript_path,
                ),
                query_vec=query_vec,
                model=model,
            )
        )

    # Concat to single
    sim_stats = pd.concat(sim_stats_list)

    # Create day columns
    sim_stats["session_date"] = pd.to_datetime(sim_stats["session_datetime"]).dt.date

    # Groupby day and get min, max, and mean
    sim_stats["day_similarity_min"] = sim_stats.groupby(["session_date"])[
        "similarity_min"
    ].transform("min")
    sim_stats["day_similarity_max"] = sim_stats.groupby(["session_date"])[
        "similarity_max"
    ].transform("max")
    sim_stats["day_similarity_mean"] = sim_stats.groupby(["session_date"])[
        "similarity_mean"
    ].transform("mean")

    return sim_stats


def compute_query_semantic_similarity_history(
    query: Union[str, List[str]],
    infrastructure_slug: Union[str, List[str]],
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    embedding_model: str = "msmarco-distilbert-base-v4",
    raise_on_error: bool = True,
    tqdm_kws: Union[Dict[str, Any], None] = None,
) -> pd.DataFrame:
    """
    Compute the semantic similarity of a query against every sentence of every meeting.
    The max, min, and mean semantic similarity of each meeting will be returned.

    Parameters
    ----------
    query: Union[str, List[str]]
        The query(ies) to compare each sentence against.
    infrastructure_slug: Union[str, List[str]]
        The CDP infrastructure(s) to connect to and pull sessions for.
    start_datetime: Optional[Union[str, datetime]]
        The earliest possible datetime for ngram history to be retrieved for.
        If provided as a string, the datetime should be in ISO format.
    end_datetime: Optional[Union[str, datetime]]
        The latest possible datetime for ngram history to be retrieved for.
        If provided as a string, the datetime should be in ISO format.
    cache_dir: Optional[Union[str, Path]]
        An optional directory path to cache the dataset. Directory is created if it
        does not exist.
        Default: "./cdp-datasets"
    embedding_model: str
        The sentence transformers model to use for embedding the query and
        each sentence.
        Default: "msmarco-distilbert-base-v4"
        All embedding models are available here:
        https://www.sbert.net/docs/pretrained-models/msmarco-v3.html
        Select any of the "Models tuned for cosine-similarity".'
    raise_on_error: bool
        Should any failure to pull files result in an error or be ignored.
        Default: True (raise on any failure)
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

    Returns
    -------
    pd.DataFrame
        The min, max, and mean semantic similarity for each event as compared
        to the query for the events within the datetime range.

    Notes
    -----
    This function requires additional dependencies.
    Install extra requirements with: `pip install cdp-data[transformers]`.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "This function requires additional dependencies. "
            "To install required extras, run `pip install cdp-data[transformers]`."
        ) from e

    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

    # Always cast query to list for easier API
    if isinstance(query, str):
        query = [query]

    # Always cast infrastructure slugs to list for easier API
    if isinstance(infrastructure_slug, str):
        infrastructure_slug = [infrastructure_slug]

    # Get semantic embedding
    model = SentenceTransformer(embedding_model)

    # Create dataframe for all histories
    semantic_histories = []

    # Start collecting datasets for each infrastructure
    for infra_slug in tqdm(
        infrastructure_slug,
        desc="Computing semantic similarity for each infrastructure",
        **tqdm_kws,
    ):
        # Get the dataset
        log.info(f"Getting session dataset for {infra_slug}")
        infra_ds = datasets.get_session_dataset(
            infrastructure_slug=infra_slug,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            store_transcript=True,
            cache_dir=cache_dir,
            raise_on_error=raise_on_error,
            tqdm_kws=tqdm_kws,
        )

        for q in tqdm(
            query,
            desc="Computing semantic similarlity for each query",
            **tqdm_kws,
        ):
            # Get query embedding
            query_vec = model.encode(q)

            # Compute semantic similarity for query
            log.info(f"Computing semantic similary history for {infra_slug}")
            infra_query_semantic_sim_history = (
                _compute_query_semantic_similarity_history(
                    data=infra_ds,
                    query_vec=query_vec,
                    model=model,
                    tqdm_kws=tqdm_kws,
                )
            )
            infra_query_semantic_sim_history["infrastructure"] = infra_slug
            infra_query_semantic_sim_history["query"] = q
            semantic_histories.append(infra_query_semantic_sim_history)

    # Convert gram histories to single dataframe
    return pd.concat(semantic_histories)
