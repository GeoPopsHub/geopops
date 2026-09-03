"""Shared pytest fixtures for the GeoPops test suite."""
import json
import os
import shutil

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DATA = os.path.join(TESTS_DIR, "data")

#: Spartanburg County, SC --- the small fixture the slow tests run against
FIXTURE_PARS = {
    "path": FIXTURE_DATA,
    "main_year": 2019,
    "geos": ["45083"],
    "commute_states": ["45", "37"],
    "use_pums": ["45", "37"],
    "random_seed": 42,
}

requires_fixture_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(FIXTURE_DATA, "processed", "acs_targets.csv")),
    reason="tests/data/processed not present; run the download+process steps first",
)


@pytest.fixture
def base_config():
    """A minimal valid config dict, not tied to any data on disk."""
    import geopops
    return geopops.make_config(path="unused", geos=["45083"], main_year=2019,
                               commute_states=["45"], use_pums=["45"], random_seed=42)


@pytest.fixture(scope="session")
def fixture_config():
    """Config pointing at the checked-in Spartanburg fixture data.

    The fixture's own ``config.json`` is used when present, because the processed
    CSVs were generated with that config's ``additional_traits``.
    """
    import geopops
    fixture_cfg_path = os.path.join(FIXTURE_DATA, "config.json")
    template = None
    if os.path.exists(fixture_cfg_path):
        with open(fixture_cfg_path) as f:
            template = json.load(f)
    return geopops.make_config(template=template, **FIXTURE_PARS)


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory):
    """A scratch copy of the fixture data, so tests never write into tests/data."""
    if not os.path.exists(os.path.join(FIXTURE_DATA, "processed")):
        pytest.skip("tests/data/processed not present")
    dest = tmp_path_factory.mktemp("geopops_run") / "data"
    shutil.copytree(FIXTURE_DATA, dest)
    shutil.rmtree(dest / "pop_export", ignore_errors=True)
    return str(dest)
