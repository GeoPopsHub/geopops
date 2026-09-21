"""Exception hierarchy for GeoPops.

Every error raised deliberately by GeoPops derives from :class:`GeoPopsError`, so
callers can catch GeoPops failures specifically::

    try:
        geopops.download_data(cfg)
    except geopops.DownloadError as e:
        ...
"""


class GeoPopsError(Exception):
    """Base class for all errors raised by GeoPops."""


class ConfigError(GeoPopsError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DownloadError(GeoPopsError):
    """A required data file could not be downloaded."""


class DataError(GeoPopsError):
    """Input data is missing, malformed, or fails a consistency check."""


class PipelineStateError(GeoPopsError):
    """A pipeline stage was run before the stage it depends on."""


__all__ = [
    "GeoPopsError",
    "ConfigError",
    "DownloadError",
    "DataError",
    "PipelineStateError",
]
