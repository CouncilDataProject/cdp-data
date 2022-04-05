#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import lru_cache

import fireo
from cdp_backend.database import models as db_models

###############################################################################


@lru_cache(1024)
def load_from_model_reference(
    model_ref: fireo.queries.query_wrapper.ReferenceDocLoader,
) -> db_models.Event:
    return model_ref.get()
