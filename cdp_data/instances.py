#!/usr/bin/env python
# -*- coding: utf-8 -*-


class CDPInstances:
    """
    Container for CDP instance infrastructure slugs.

    Examples
    --------
    >>> from cdp_data import datasets, CDPInstances
    ... ds = datasets.get_session_dataset(
    ...     infrastructure_slug=CDPInstances.Seattle
    ... )
    """

    Seattle = "cdp-seattle-21723dcf"
    KingCounty = "cdp-king-county-b656c71b"
    Portland = "cdp-portland-d2bbda97"
    Missoula = "missoula-council-data-proj"
