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
    Denver = "cdp-denver-962aefef"
    Alameda = "cdp-alameda-d3dabe54"
    Boston = "cdp-boston-c384047b"
    Oakland = "cdp-oakland-ba81c097"
    Charlotte = "cdp-charlotte-98a7c348"
    SanJose = "cdp-san-jose-5d9db455"
    MountainView = "cdp-mountain-view-7c8a47df"
    Milwaukee = "cdp-milwaukee-9f60e352"
    LongBeach = "cdp-long-beach-49323fe9"
