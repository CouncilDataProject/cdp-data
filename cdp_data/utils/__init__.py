"""Utils package for cdp_data."""

from gcsfs import GCSFileSystem

from .db_utils import connect_to_database
from .fs_utils import connect_to_filestore


def connect_to_infrastructure(infrastructure_slug: str) -> GCSFileSystem:
    """
    Simple function to shorten how many imports and code it takes to connect
    to a CDP infrastructure.
    """
    connect_to_database(infrastructure_slug)
    return connect_to_filestore(infrastructure_slug)
