"""End-to-end tests of the GeoPops pipeline.

These need the processed fixture data in ``tests/data/processed`` and take minutes,
so they are marked ``slow``. Run the fast suite with::

    pytest -m "not slow"
"""
import os

import pytest

import geopops
from conftest import requires_fixture_data

pytestmark = [pytest.mark.slow, requires_fixture_data]


@pytest.mark.network
@pytest.mark.skip(reason="Manual run only: downloads several GB from Census/LODES")
def test_download(fixture_config):
    return geopops.download_data(fixture_config)


@pytest.mark.skip(reason="Manual run only: rebuilds the processed fixture (~5 min)")
def test_processing(fixture_config):
    return geopops.process_data(fixture_config)


@pytest.fixture(scope="module")
def pop(fixture_config, run_dir):
    """One generated population, shared by the tests below."""
    cfg = dict(fixture_config)
    cfg["path"] = run_dir
    cfg["CO_maxgens"] = 5000     # enough to converge on most CBGs, fast enough to test
    return geopops.generate_pop(cfg, seed=42, verbose=0)


def test_co_assigns_households_to_every_cbg(pop):
    assert pop.co_results
    for county, cbgs in pop.co_results.items():
        assert cbgs, f"county {county} got no CBGs"
        for cbg, serials in cbgs.items():
            assert serials, f"CBG {cbg} got no households"


def test_synthpop_populates_people_and_households(pop):
    assert len(pop.people) > 0
    assert len(pop.households) > 0
    # every household member must resolve to a real person
    for hh in pop.households.values():
        for pkey in hh.people:
            assert pkey in pop.people


def test_people_carry_the_configured_traits(pop, fixture_config):
    expected = list(fixture_config["additional_traits"])
    person = next(iter(pop.people.values()))
    assert list(person.schema.names) == expected
    assert set(person.traits) == set(expected)


def test_networks_are_symmetric_and_sized_to_the_population(pop):
    n = len(pop.adj_mat_keys)
    for name in ("adj_hh", "adj_sch", "adj_wp", "adj_gq"):
        m = getattr(pop, name)
        assert m.shape == (n, n), name
        assert (m != m.T).nnz == 0, f"{name} is not symmetric"


def test_export_writes_every_expected_file(pop):
    expected = [
        "cbg_idxs.csv", "hh.csv", "people.csv", "sch_students.csv", "sch_workers.csv",
        "gqs.csv", "gq_residents.csv", "gq_workers.csv", "company_workers.csv",
        "outside_workers.csv", "adj_mat_keys.csv", "adj_dummy_keys.csv",
        "adj_out_workers.csv", "adj_upper_triang_hh.mtx", "adj_upper_triang_sch.mtx",
        "adj_upper_triang_wp.mtx", "adj_upper_triang_gq.mtx",
        "adj_upper_triang_non_hh.mtx",
    ]
    for name in expected:
        path = os.path.join(pop.pop_export_dir, name)
        assert os.path.exists(path), f"missing {name}"
        assert os.path.getsize(path) > 0, f"empty {name}"


def test_exported_people_columns_follow_the_config(pop, fixture_config):
    import pandas as pd
    df = pd.read_csv(os.path.join(pop.pop_export_dir, "people.csv"))
    for trait in fixture_config["additional_traits"]:
        assert trait in df.columns
    assert len(df) == len(pop.people)


def test_starsim_people_and_networks(pop):
    ppl = geopops.to_starsim_people(pop.pop_export_dir, verbose=0)
    assert len(ppl) == len(pop.adj_mat_keys)

    nets = geopops.starsim_networks(pop.pop_export_dir, seed=42, save=False, verbose=0)
    assert len(nets) == 4
    assert all(len(n.edges.p1) > 0 for n in nets)


def test_two_populations_do_not_share_cached_networks(pop, tmp_path):
    """Regression: ForStarsim cached networks in class state and never invalidated."""
    a = geopops.starsim_network("homenet", pop.pop_export_dir, save=False)
    b = geopops.starsim_network("schoolnet", pop.pop_export_dir, save=False)
    assert len(a.edges.p1) != len(b.edges.p1) or not (a.edges.p1 == b.edges.p1).all()
