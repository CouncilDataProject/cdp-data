"""Top-level package for cdp_data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cdp-data")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown, Council Data Project Contributors"
__email__ = "evamaxfieldbrown@gmail.com"

from .instances import CDPInstances

__all__ = ["CDPInstances"]
