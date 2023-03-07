#!/usr/bin/env python

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Union

import fireo
import pandas as pd
from dataclasses_json import dataclass_json
from fireo.models import Model
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

###############################################################################


@dataclass
class _ModelRefJoiner:
    join_id: str
    model_ref: fireo.queries.query_wrapper.ReferenceDocLoader


@dataclass_json
@dataclass
class _ModelJoiner:
    join_id: str
    model: Model


###############################################################################


def connect_to_database(infrastructure_slug: str) -> None:
    """
    Simple function to shorten how many imports and code it takes to connect
    to a CDP database.
    """
    fireo.connection(
        client=Client(
            project=infrastructure_slug,
            credentials=AnonymousCredentials(),
        )
    )


@lru_cache(1024)
def load_from_model_reference(
    model_ref: fireo.queries.query_wrapper.ReferenceDocLoader,
) -> Model:
    """
    Load a CDP database model from a ReferenceDocLoader or potentially from cache.

    Parameters
    ----------
    model_ref: fireo.queries.query_wrapper.ReferenceDocLoader
        The model reference to load.

    Returns
    -------
    model: Model
        The loaded (retrieved from database or cache) CDP database model.

    See Also
    --------
    cdp_data.utils.db_utils.load_model_from_reference_joiner
    cdp_data.utils.db_utils.load_model_from_pd_columns

    Notes
    -----
    LRU cache size is 1024 items.
    """
    return model_ref.get()


def _load_model_from_reference_joiner(
    ref_joiner: _ModelRefJoiner,
) -> _ModelJoiner:
    """
    Load a CDP database model from a ModelRefJoiner.

    Parameters
    ----------
    ref_joiner: ModelRefJoiner
        The join id string and the model ref to load.

    Returns
    -------
    model_joiner: ModelJoiner
        The join id and the loaded model.

    See Also
    --------
    cdp_data.utils.db_utils.load_from_model_reference
    cdp_data.utils.db_utils.load_model_from_pd_columns

    Notes
    -----
    This function is primarily intended for use with pandas DataFrame joins where
    you may want to load a referenced model that is a column value and join back to
    the original DataFrame.

    Additionally, this function uses the
    `cdp_data.utils.db_utils.load_from_model_reference` function to load the full
    database model which itself uses an LRU cache.
    """
    return _ModelJoiner(
        join_id=ref_joiner.join_id,
        model=load_from_model_reference(ref_joiner.model_ref),
    )


def load_model_from_pd_columns(
    data: pd.DataFrame,
    join_id_col: str,
    model_ref_col: str,
    drop_original_model_ref: bool = True,
    tqdm_kws: Union[Dict[str, Any], None] = None,
) -> pd.DataFrame:
    """
    Load a model reference and attach the loaded model back to the original DataFrame.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame which contains a model ReferenceDocLoader to fetch and reattach
        the loaded model to.
    join_id_col: str
        The column name to use for joining the original provided DataFrame to the
        loaded models DataFrame.
    model_ref_col: str
        The column name which contains the model ReferenceDocLoader objects.
    drop_original_model_ref: bool
        After loading and joining all models to the DataFrame, should the original
        `model_ref_col` be dropped.
        Default: True (drop the original `model_ref_column`)
    tqdm_kws: Dict[str, Any]
        A dictionary with extra keyword arguments to provide to tqdm progress
        bars. Must not include the `desc` keyword argument.

    Returns
    -------
    data: pd.DataFrame
        A DataFrame with all of the original data and all the models loaded from the
        original DataFrame's `model_ref_col` ReferenceDocLoader objects.

    See Also
    --------
    cdp_data.utils.db_utils.load_from_model_reference
    cdp_data.utils.db_utils.load_model_from_pd_columns

    Notes
    -----
    This function loads all models using a threadpool. Because of this threading,
    the order of the rows may be different from the original DataFrame to the result
    DataFrame.

    Additionally, this function utilizes an LRU cache during model loading.

    Examples
    --------
    Fetch sessions from a CDP database and then fetch and attach all
    referenced events to each session.

    >>> from cdp_backend.database import models as db_models
    ... from cdp_data.utils import db_utils
    ... import pandas as pd
    ... # Connect, fetch sessions and unpack, threaded event attachment to session df
    ... db_utils.connect_to_database("cdp-seattle-21723dcf")
    ... sessions = pd.DataFrame([
    ...     s.to_dict() for s in db_models.Session.collection.fetch()
    ... ])
    ... # Fetch all models in the `event_ref` column and join on session id
    ... event_attached = db_utils.load_model_from_pd_columns(
    ...     sessions,
    ...     join_id_col="id",
    ...     model_ref_col="event_ref",
    ... )
    """
    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

    # Get models
    loaded_models = thread_map(
        _load_model_from_reference_joiner,
        [
            _ModelRefJoiner(
                join_id=row[join_id_col],
                model_ref=row[model_ref_col],
            )
            for _, row in data.iterrows()
        ],
        desc=f"Fetching each model attached to {model_ref_col}",
        **tqdm_kws,
    )

    # Convert to dataframe
    models_to_join = pd.DataFrame([j.to_dict() for j in loaded_models])

    # Rename column to collection name
    models_to_join = models_to_join.rename(
        {"model": models_to_join.loc[0].model.collection_name},
        axis=1,
    )

    # Join and return
    joined = data.join(models_to_join.set_index("join_id"), on=join_id_col)

    # Handle model ref drop
    if drop_original_model_ref:
        joined = joined.drop([model_ref_col], axis=1)

    return joined


def expand_models_from_pd_column(
    data: pd.DataFrame,
    model_col: str,
    model_attr_rename_lut: Dict[str, str],
    tqdm_kws: Union[Dict[str, Any], None] = None,
) -> pd.DataFrame:
    # Handle default dict
    if not tqdm_kws:
        tqdm_kws = {}

    # Store individual rows
    expanded_data: List[pd.Series] = []

    # Iter rows and unpack
    for _, row in tqdm(
        data.iterrows(),
        desc=f"Expanding {model_col} models",
        **tqdm_kws,
    ):
        for model_attr_name, attr_replace_name in model_attr_rename_lut.items():
            row[attr_replace_name] = getattr(row[model_col], model_attr_name)

        expanded_data.append(row)

    # New dataframe with expanded data
    return pd.DataFrame(expanded_data)
