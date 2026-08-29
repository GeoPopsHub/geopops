"""Unit tests for contact network construction."""
import numpy as np
import pytest

from geopops import networks


def keys(n, groups=None):
    """n person keys of the form (p_id, hh_id, cbg_id, group_label)."""
    groups = groups or ["g0"] * n
    return [(i, 1, 1, groups[i]) for i in range(n)]


class TestConnectComplete:
    def test_edge_count(self):
        assert len(networks.connect_complete(keys(5))) == 10  # 5 choose 2

    @pytest.mark.parametrize("n", [0, 1])
    def test_too_small_to_connect(self, n):
        assert networks.connect_complete(keys(n)) == []

    def test_deduplicates_keys(self):
        k = keys(3)
        assert len(networks.connect_complete(k + k)) == 3


class TestConnectSmallWorld:
    def test_below_min_n_is_complete(self):
        edges = networks.connect_small_world(keys(4), K=4, min_N=10, B=0.25,
                                             rng=np.random.default_rng(0))
        assert len(edges) == 6

    def test_mean_degree_near_k(self):
        n, K = 200, 8
        edges = networks.connect_small_world(keys(n), K=K, min_N=10, B=0.25,
                                             rng=np.random.default_rng(0))
        assert 2 * len(edges) / n == pytest.approx(K, abs=1)

    def test_deterministic_given_a_seed(self):
        a = networks.connect_small_world(keys(50), 6, 10, 0.25, np.random.default_rng(1))
        b = networks.connect_small_world(keys(50), 6, 10, 0.25, np.random.default_rng(1))
        assert a == b

    def test_no_self_loops(self):
        edges = networks.connect_small_world(keys(60), 6, 10, 0.25, np.random.default_rng(2))
        assert all(u != v for u, v in edges)


class TestConnectSBM:
    def test_below_min_n_is_complete(self):
        edges = networks.connect_SBM(keys(4), K=8, min_N=10, assoc_coeff=0.9,
                                     rng=np.random.default_rng(0))
        assert len(edges) == 6

    def test_no_isolated_nodes(self):
        k = keys(100, [f"g{i % 3}" for i in range(100)])
        edges = networks.connect_SBM(k, K=6, min_N=8, assoc_coeff=0.9,
                                     rng=np.random.default_rng(0))
        connected = {u for u, _ in edges} | {v for _, v in edges}
        assert len(connected) == 100

    def test_assortativity_concentrates_edges_within_groups(self):
        k = keys(200, [f"g{i % 2}" for i in range(200)])
        rng = np.random.default_rng(0)
        high = networks.connect_SBM(k, 8, 10, assoc_coeff=1.0, rng=rng)
        low = networks.connect_SBM(k, 8, 10, assoc_coeff=0.0, rng=rng)

        def within(edges):
            return sum(u[3] == v[3] for u, v in edges) / max(len(edges), 1)

        assert within(high) > within(low)
        assert within(high) > 0.95

    def test_single_group(self):
        edges = networks.connect_SBM(keys(50), 6, 10, 0.9, use_groups=False,
                                     rng=np.random.default_rng(0))
        assert len(edges) > 0


class TestSpFromGroups:
    def test_matrix_is_symmetric_and_boolean(self):
        p_idxs = {(i, 1, 1): i for i in range(5)}
        m = networks.sp_from_groups(networks.connect_complete, [keys(5)], p_idxs)
        assert m.shape == (5, 5)
        assert m.dtype == bool
        assert (m != m.T).nnz == 0

    def test_unknown_keys_are_dropped(self):
        p_idxs = {(0, 1, 1): 0, (1, 1, 1): 1}   # key (2,1,1) is absent
        m = networks.sp_from_groups(networks.connect_complete, [keys(3)], p_idxs)
        assert m.shape == (2, 2) and m.nnz == 2

    def test_empty_input(self):
        m = networks.sp_from_groups(networks.connect_complete, [], {(0, 1, 1): 0})
        assert m.nnz == 0
