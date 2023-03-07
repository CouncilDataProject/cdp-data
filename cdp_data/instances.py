#!/usr/bin/env python


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
    Albuquerque = "cdp-albuquerque-1d29496e"
    Richmond = "cdp-richmond-a3d06941"
    Louisville = "cdp-louisville-6fd32a38"
    Atlanta = "cdp-atlanta-37e7dd70"
