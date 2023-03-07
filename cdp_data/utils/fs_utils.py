#!/usr/bin/env python

from gcsfs import GCSFileSystem

###############################################################################


def connect_to_filestore(infrastructure_slug: str) -> GCSFileSystem:
    """
    Simple function to shorten how many imports and code it takes to connect
    to a CDP file store.
    """
    return GCSFileSystem(project=infrastructure_slug, token="anon")
