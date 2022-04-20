# -*- coding: utf-8 -*-

"""Top-level package for cdp_data."""

__author__ = "Jackson Maxfield Brown"
__email__ = "jmaxfieldbrown@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.4"


from .instances import CDPInstances  # noqa: F401


def get_module_version() -> str:
    return __version__
