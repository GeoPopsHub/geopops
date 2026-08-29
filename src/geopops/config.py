"""Configuration loading, merging, and validation for GeoPops.

The packaged ``config.json`` is a read-only *template*. A run's config is an
ordinary dict, and :func:`make_config` returns one; persisting it is the caller's
choice and, by default, writes into the run's own output directory rather than into
the installed package.

Secrets (the Census API key) come from the environment or a ``.env`` file and are
never written into the packaged template.
"""
import json
import os
import warnings

from dotenv import load_dotenv, find_dotenv

from .exceptions import ConfigError

load_dotenv(find_dotenv())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "config.json")

#: Config keys that must never be written into the packaged template
SENSITIVE_CONFIG_KEYS = ("census_api_key",)

#: Config keys that may be supplied as user-facing overrides
OVERRIDE_KEYS = ("census_api_key", "main_year", "geos", "commute_states",
                 "use_pums", "path", "random_seed")

#: ACS / decennial tables required by the pipeline, used to backfill minimal configs
DEFAULT_ACS_REQUIRED = [
    "B01001", "B09019", "B09020", "C24030", "B23025", "C24010", "B11016",
    "B11012", "B23009", "B11004", "B19001", "B22010", "B09021", "B09018",
    "B11001H", "B11001I", "B25006",
]
DEFAULT_DEC_REQUIRED = ["P43", "P18"]


def _merge_dict(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def load_config(base_dir=None, path=None):
    """Load a config file, applying any sibling ``config.local.json`` overrides.

    Args:
        base_dir: directory holding ``config.json``. Defaults to the package
            directory (the shipped template).
        path: explicit path to a config file; overrides `base_dir`.

    Returns:
        dict: the config.
    """
    if path is not None:
        cfg_path = path
        cfg_dir = os.path.dirname(os.path.abspath(path))
    else:
        cfg_dir = base_dir if base_dir is not None else BASE_DIR
        cfg_path = os.path.join(cfg_dir, "config.json")

    if not os.path.exists(cfg_path):
        raise ConfigError(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)

    # Optional untracked local overrides for machine-specific values.
    local_cfg_path = os.path.join(cfg_dir, "config.local.json")
    if os.path.exists(local_cfg_path):
        with open(local_cfg_path) as f:
            config = _merge_dict(config, json.load(f))

    return config


def _template_config(config):
    """A copy safe to ship as the package template (no secrets)."""
    template = dict(config)
    for key in SENSITIVE_CONFIG_KEYS:
        template[key] = None
    return template


def save_config(config, config_path=None, *, sanitize=False):
    """Write `config` as JSON.

    Args:
        config: the config dict.
        config_path: destination path, or a directory to write ``config.json`` into.
            Defaults to the run's own ``path`` directory.
        sanitize: blank out secrets first (used only for the packaged template).
    """
    if config_path is None:
        config_path = os.path.join(config.get("path", "."), "config.json")
    elif os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")

    parent = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(parent, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(_template_config(config) if sanitize else config, f, indent=4)
    return config_path


def compute_decennial_year(main_year):
    """The decennial census vintage that applies to `main_year`."""
    try:
        year = int(main_year)
    except (TypeError, ValueError):
        raise ConfigError(
            f"main_year must be an integer year, got {main_year!r}."
        ) from None
    return 2020 if year >= 2020 else 2010


def update_config_values(config, **overrides):
    """Apply user overrides to `config` in place, falling back to the environment.

    Only keys in :data:`OVERRIDE_KEYS` are accepted; anything else is a typo and is
    reported rather than silently ignored. ``None`` values mean "leave unchanged".
    """
    unknown = set(overrides) - set(OVERRIDE_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown config override(s): {sorted(unknown)}. "
            f"Valid overrides: {list(OVERRIDE_KEYS)}"
        )

    # Sensitive/user-specific values fall back to the environment
    if overrides.get("census_api_key") is None:
        overrides["census_api_key"] = os.environ.get("CENSUS_API_KEY")

    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if overrides.get("main_year") is not None:
        config["decennial_year"] = compute_decennial_year(overrides["main_year"])

    config.setdefault("acs_required", list(DEFAULT_ACS_REQUIRED))
    config.setdefault("dec_required", list(DEFAULT_DEC_REQUIRED))
    return config


def validate_config(config):
    """Check a config for the mistakes that otherwise surface deep in the pipeline.

    Raises:
        ConfigError: if a required key is missing or a value is unusable.
    """
    for key in ("path", "main_year", "geos"):
        if not config.get(key):
            raise ConfigError(f"config is missing required key {key!r}.")

    if not isinstance(config["geos"], list | tuple) or not config["geos"]:
        raise ConfigError("config['geos'] must be a non-empty list of state/county FIPS codes.")

    compute_decennial_year(config["main_year"])  # raises if unparseable

    # Traits are carried generically, but only PUMS-derived ones will have values,
    # and CO does not target them --- worth saying once, up front.
    traits = config.get("additional_traits") or []
    if not isinstance(traits, list | tuple):
        raise ConfigError("config['additional_traits'] must be a list of column names.")

    if config.get("random_seed") is None:
        warnings.warn(
            "No random_seed set: this run will not be reproducible. "
            "Pass seed=... or set config['random_seed'].",
            stacklevel=3,
        )
    return config


def make_config(path=None, *, base_dir=None, template=None, save=False, **overrides):
    """Build a run configuration from the packaged template plus overrides.

    Args:
        path: output directory for the run (also where results are written).
        base_dir: directory to load the template from; defaults to the package.
        template: a full config dict to use instead of loading a template.
        save: if True, also write ``<path>/config.json``.
        **overrides: any of :data:`OVERRIDE_KEYS`.

    Returns:
        dict: the effective config.

    Example::

        cfg = geopops.make_config(path="data", geos=["45083"], main_year=2019,
                                  commute_states=["45", "37"], use_pums=["45", "37"])
    """
    config = dict(template) if template is not None else load_config(base_dir)
    if path is not None:
        overrides["path"] = path
    update_config_values(config, **overrides)
    validate_config(config)
    if save:
        save_config(config)
    return config
