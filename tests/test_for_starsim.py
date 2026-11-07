"""Test the Starsim bridge on a tiny synthetic population, so no downloaded data is needed."""
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse
import starsim as ss
import starsim.library as ssl

from geopops import for_starsim

N_HOUSEHOLD = 90
N_GQ = 30 # Group-quarters residents: in the keys, but not in people.csv
N = N_HOUSEHOLD + N_GQ
RACES = ['race_white_alone', 'race_black_alone', 'race_amerindian_or_alaskan', 'race_asian_alone',
         'race_pacific_alone', 'race_other_alone', 'race_two_or_more', 'hispanic']


@pytest.fixture
def pop(tmp_path):
    """Write the parts of pop_export that for_starsim reads."""
    out = tmp_path / 'pop_export'
    out.mkdir()
    p_id = np.arange(1, N + 1)
    in_hh = p_id <= N_HOUSEHOLD
    hh_id = np.where(in_hh, (p_id - 1) // 3 + 1, 0)
    cbg_id = np.where(in_hh, 1, 0)
    keys = pd.DataFrame(dict(index_one=p_id, index_zero=p_id - 1, p_id=p_id, hh_id=hh_id, cbg_id=cbg_id))
    keys.to_csv(out / 'adj_mat_keys.csv', index=False)

    people = keys[in_hh].drop(columns=['index_one', 'index_zero']).assign(
        sample_index=1, age=np.arange(N_HOUSEHOLD) % 95, female=np.arange(N_HOUSEHOLD) % 2, working=0,
        commuter=0, commuter_income_category=np.nan, commuter_workplace_category=np.nan,
        **{race: 0 for race in RACES})
    people.to_csv(out / 'people.csv', index=False)
    pd.DataFrame(dict(p_id=[1], hh_id=[1], cbg_id=[1], sch_grade=['k'], sch_code=[7])).to_csv(out / 'sch_students.csv', index=False)
    pd.DataFrame(dict(cbg_id=[1], cbg_geocode=[240270001001])).to_csv(out / 'cbg_idxs.csv', index=False)
    pd.DataFrame(dict(cbg_id=[1] * 30, hh_id=np.arange(1, 31), sample_index=1)).to_csv(out / 'hh.csv', index=False)

    ring = scipy.sparse.coo_matrix((np.ones(N - 1), (np.arange(1, N), np.arange(N - 1))), shape=(N, N))
    for tag in ['hh', 'sch', 'wp', 'gq']:
        scipy.io.mmwrite(out / f'adj_upper_triang_{tag}.mtx', ring)
    return str(tmp_path)


def make_sim(pop, disease, **kwargs):
    ppl = for_starsim.people(path=pop)
    networks = [for_starsim.network(name, path=pop) for name in for_starsim._LAYERS]
    return ss.Sim(people=ppl, networks=networks, diseases=disease, start=2020, stop=2021, dt=ss.days(7), verbose=0, **kwargs)


def test_people_values(pop):
    """Agents keep the population's values, including through the saved file."""
    for ppl in [for_starsim.people(path=pop), ss.load(f'{pop}/pop_export/starsim/ppl.pkl')]:
        people = ss.Sim(people=ppl).init().people
        assert np.array_equal(people.age.raw[:N_HOUSEHOLD], np.arange(N_HOUSEHOLD) % 95)
        assert np.isnan(people.age.raw[N_HOUSEHOLD:N]).all()
        assert np.array_equal(people.female.raw[:N_HOUSEHOLD], np.arange(N_HOUSEHOLD) % 2 == 1)
        assert not people.female.raw[N_HOUSEHOLD:N].all() # A missing sex isn't cast to True
        assert people.state.raw[0] == 24
        assert (people.state.raw[N_HOUSEHOLD:N] == 0).all() # No block group, so no state
        assert (people.county.raw[:N_HOUSEHOLD] == 24027).all()
        assert (people.tract.raw[:N_HOUSEHOLD] == 24027000100).all()


def test_leading_zero_state(pop):
    """A geocode saved as a number loses a leading zero, which mustn't change the state."""
    pd.DataFrame(dict(cbg_id=[1], cbg_geocode=[10010201001])).to_csv(f'{pop}/pop_export/cbg_idxs.csv', index=False)
    people = ss.Sim(people=for_starsim.people(path=pop)).init().people
    assert people.state.raw[0] == 1
    assert people.county.raw[0] == 1001


def test_missing_sex_uses_observed_share():
    female = np.r_[np.ones(300), np.zeros(700), np.full(100_000, np.nan)]
    filled = for_starsim._fill_sex(female)
    assert filled.dtype == bool
    assert filled.mean() == pytest.approx(0.3, abs=0.01)
    assert np.array_equal(filled, for_starsim._fill_sex(female)) # The same export every time


@pytest.mark.parametrize('disease, outcome', [
    (ssl.AgentSIS(init_prev=0.5, beta=ss.peryear(1)), 'infected'),
    (ss.SIS(init_prev=0.5, beta=ss.peryear(1)), 'I'),
])
def test_tracking(pop, disease, outcome):
    """The analyzer works with diseases written as code and declared, on all four networks."""
    tracker = for_starsim.subgroup_tracking('agegroup', outcome, name='tracker')
    sim = make_sim(pop, disease, analyzers=tracker).run()
    assert sim.pars.networks is not None and len(sim.networks) == 4
    assert [len(net) for net in sim.networks.values()] == [N - 1] * 4
    counts = sim.analyzers.tracker.get_subgroup_data().drop(columns='agegroup')
    assert counts.shape[1] == sim.t.npts
    assert counts['t_0'].sum() == sim.diseases[0].results[outcome][0] # The state the run starts from
