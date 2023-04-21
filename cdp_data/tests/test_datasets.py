#!/usr/bin/env python

from fireo.models import Model
from numpy.testing import assert_raises

from cdp_data import CDPInstances, datasets

###############################################################################


def test_get_session_dataset() -> None:
    ds = datasets.get_session_dataset(
        CDPInstances.Seattle,
        start_datetime="2022-04-04",
        end_datetime="2022-04-06",
        store_video=True,
        store_audio=True,
        store_transcript=True,
        store_transcript_as_csv=True,
        replace_py_objects=True,
    )
    assert len(ds) == 3

    # Assert that no columns are objects
    for col in ds.columns:
        assert not isinstance(ds.loc[0][col], Model)

    # Store
    datasets.save_dataset(ds, "test-outputs-session-dataset.csv")
    datasets.save_dataset(ds, "test-outputs-session-dataset.parquet")
    with assert_raises(ValueError):
        datasets.save_dataset(ds, "blah.bad-suffix")


def test_get_vote_dataset() -> None:
    ds = datasets.get_vote_dataset(
        CDPInstances.Seattle,
        start_datetime="2022-10-01",
        end_datetime="2022-10-10",
        replace_py_objects=True,
    )
    assert len(ds) == 207

    # Assert that no columns are objects
    for col in ds.columns:
        assert not isinstance(ds.loc[0][col], Model)

    # Store
    datasets.save_dataset(ds, "test-outputs-vote-dataset.csv")
    datasets.save_dataset(ds, "test-outputs-vote-dataset.parquet")
    with assert_raises(ValueError):
        datasets.save_dataset(ds, "blah.bad-suffix")
