"""
Tests for the pure-Python GeoPops pipeline (RunPython).

Prerequisites:
- Preprocessed data in tests/data/processed/ (or run this script with redownload=True)
"""

import sciris as sc
import geopops
import pytest

pars_geopops = {'path': "data",
                'main_year': 2019,
                'geos': ["45083"],
                'commute_states': ["45","37"],
                'use_pums': ["45","37"],
                }

c = geopops.WriteConfig(**pars_geopops)


@sc.timer()
@pytest.mark.skip(reason="Manual run only (too slow for automated tests)")
def test_download():
    """ Check that download works (~10 min)"""
    d = geopops.DownloadData(auto_run=True)
    return d


@sc.timer()
@pytest.mark.skip(reason="Manual run only (too slow for automated tests)")
def test_processing():
    """ Check that data processing works (~5 min)"""
    p = geopops.ProcessData(auto_run=True) # auto_run=True to run all 
    return p


@pytest.fixture(scope="module")
def runner():
    """Shared RunPython instance for sequential pipeline tests."""
    r = geopops.RunPython()
    return r


@sc.timer()
def test_python_CO(runner):
    """Test combinatorial optimization in Python."""
    runner.CO()
    assert runner.co_results is not None
    assert len(runner.co_results) > 0
    for county, cbg_dict in runner.co_results.items():
        assert len(cbg_dict) > 0
        for cbg, serials in cbg_dict.items():
            assert len(serials) > 0
    return


@sc.timer()
def test_python_synthpop(runner):
    """Test synthetic population generation in Python."""
    runner.SynthPop()
    assert runner.people is not None
    assert runner.households is not None
    assert len(runner.people) > 0
    assert len(runner.households) > 0
    return


@sc.timer()
def test_python_export(runner):
    """Test export and ForStarsim integration."""
    runner.Export()
    ppl = geopops.ForStarsim.People()
    h = geopops.ForStarsim.GPNetwork(name='homenet', edge_weight=1.0)
    s = geopops.ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
    w = geopops.ForStarsim.GPNetwork(name='worknet', edge_weight=1.0)
    g = geopops.ForStarsim.GPNetwork(name='gqnet', edge_weight=1.0)
    return ppl, h, s, w, g


if __name__ == "__main__":
    T = sc.timer()

    # Download & process data files
    redownload = False
    if redownload:
        test_download()
        test_processing()  

    # Run GeoPops on the data
    r = geopops.RunPython()
    test_python_CO(r)
    test_python_synthpop(r)
    outputs = test_python_export(r)

    T.toc()
