"""Unit tests for the combinatorial optimization (simulated annealing) step."""
import numpy as np
import pytest

from geopops import co


@pytest.fixture
def pool():
    """A synthetic sample pool and a target drawn from it, so a good fit exists."""
    rng = np.random.default_rng(0)
    samples = rng.integers(0, 4, size=(500, 12)).astype(np.int64)
    idxs = np.arange(500)
    targ = samples[rng.integers(0, 500, 40)].sum(axis=0, keepdims=True)
    return samples, idxs, targ


PARAMS = dict(maxgens=2000, critval=-1.0, cooldown=0.99)  # never exit early


class TestFTdist:
    def test_zero_for_identical(self):
        v = np.array([[1, 2, 3]])
        assert co.FTdist(v, v) == 0.0

    def test_positive_and_symmetric(self):
        a, b = np.array([[1, 2, 3]]), np.array([[4, 0, 3]])
        assert co.FTdist(a, b) > 0
        assert co.FTdist(a, b) == pytest.approx(co.FTdist(b, a))


class TestAnneal:
    def test_returns_requested_number_of_samples(self, pool):
        samples, idxs, targ = pool
        result, gens, score, temp = co.anneal(samples, idxs, targ, 40, PARAMS,
                                              np.random.default_rng(1))
        assert len(result) == 40
        assert set(result).issubset(set(idxs))
        assert gens > 0 and np.isfinite(score)

    def test_improves_on_the_starting_fit(self, pool):
        samples, idxs, targ = pool
        rng = np.random.default_rng(1)
        start = samples[rng.integers(0, len(samples), 40)].sum(axis=0, keepdims=True)
        start_score = co.FTdist(start, targ)
        _, _, score, _ = co.anneal(samples, idxs, targ, 40, PARAMS, np.random.default_rng(1))
        assert score < start_score

    def test_deterministic_given_a_seed(self, pool):
        samples, idxs, targ = pool
        a = co.anneal(samples, idxs, targ, 40, PARAMS, np.random.default_rng(3))
        b = co.anneal(samples, idxs, targ, 40, PARAMS, np.random.default_rng(3))
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]

    def test_incremental_summary_matches_a_full_recompute(self, pool):
        """The inner loop updates its running sum incrementally; check it stays exact."""
        samples, idxs, targ = pool
        result, _, score, _ = co.anneal(samples, idxs, targ, 40, PARAMS,
                                        np.random.default_rng(5))
        recomputed = co.FTdist(samples[result].sum(axis=0, keepdims=True), targ)
        assert score == pytest.approx(recomputed, abs=1e-9)

    def test_empty_pool(self):
        result, gens, score, temp = co.anneal(
            np.zeros((0, 5), np.int64), np.array([], int), np.zeros((1, 5), np.int64),
            10, PARAMS, np.random.default_rng(0))
        assert len(result) == 0 and gens == 0 and score == float("inf")

    def test_stops_at_critval(self, pool):
        samples, idxs, targ = pool
        params = dict(maxgens=10**6, critval=1e9, cooldown=0.99)
        _, gens, _, _ = co.anneal(samples, idxs, targ, 40, params, np.random.default_rng(0))
        assert gens == 1  # the criterion is met immediately

    def test_respects_maxgens(self, pool):
        samples, idxs, targ = pool
        _, gens, _, _ = co.anneal(samples, idxs, targ, 40, dict(PARAMS, maxgens=50),
                                  np.random.default_rng(0))
        assert gens == 51


class TestUrbanizationLookup:
    def test_urban(self):
        assert co.urbanization_lookup(np.array([1.0, 0.5, 0.0]), 1.0).tolist() == [True, False, False]

    def test_rural(self):
        assert co.urbanization_lookup(np.array([1.0, 0.5, 0.0]), 0.0).tolist() == [False, False, True]

    def test_mixed_uses_a_window(self):
        got = co.urbanization_lookup(np.array([0.5, 0.55, 0.9]), 0.5)
        assert got.tolist() == [True, True, False]

    def test_nan_never_matches(self):
        assert not co.urbanization_lookup(np.array([np.nan]), 0.5).any()
        assert not co.urbanization_lookup(np.array([np.nan]), 1.0).any()


class TestSampleLookup:
    def test_shares_index_arrays_between_equal_targets(self):
        """Equal targets must share one array so callers can cache the sub-pool."""
        import pandas as pd
        df = pd.DataFrame({"st_puma": ["a", "b", "a", "c"]})
        out = co.sample_lookup(df, "st_puma", ["a", "b", "a"])
        assert [k for k, _ in out] == ["a", "b", "a"]
        assert out[0][1] is out[2][1]
        assert out[0][1].tolist() == [0, 2]

    def test_missing_values_never_match(self):
        import pandas as pd
        df = pd.DataFrame({"county": ["01", None, "01"]})
        (_, idx), = co.sample_lookup(df, "county", ["01"])
        assert idx.tolist() == [0, 2]

    def test_absent_target_gives_empty(self):
        import pandas as pd
        df = pd.DataFrame({"county": ["01"]})
        (_, idx), = co.sample_lookup(df, "county", ["99"])
        assert idx.tolist() == []
