#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    )
    assert len(ds) == 2
