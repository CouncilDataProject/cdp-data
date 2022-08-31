#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Literal
import pandas as pd

###############################################################################


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


ALL_INSTANCES = [
    getattr(inst) for inst in dir(CDPInstances) if not inst.startswith("__")
]

###############################################################################

class CDPInstanceDataChecks:
    existance: str = "existance"
    count: str = "count"

def check_instance_data_availability(
    instances: List[str] = ALL_INSTANCES,
    check: str = CDPInstanceDataChecks.existance,
) -> pd.DataFrame:
    pass
