"""Unit tests for the pure helper functions in geopops.utils.

These need no data and run in well under a second.
"""
import numpy as np
import pytest

from geopops import utils


class TestLrRound:
    """Largest-remainder rounding preserves the total."""

    def test_preserves_sum(self):
        v = np.array([1.4, 1.4, 1.4, 1.4, 1.4])
        out = utils.lrRound(v)
        assert out.sum() == round(v.sum())
        assert out.dtype == np.int64

    def test_gives_extra_to_largest_remainders(self):
        # 0.9 has the largest remainder, so it rounds up first
        out = utils.lrRound(np.array([0.9, 0.6, 0.5]))
        assert out.tolist() == [1, 1, 0]

    def test_integers_unchanged(self):
        v = np.array([3.0, 1.0, 6.0])
        assert utils.lrRound(v).tolist() == [3, 1, 6]

    def test_zeros(self):
        assert utils.lrRound(np.zeros(4)).tolist() == [0, 0, 0, 0]

    @pytest.mark.parametrize("n", [1, 5, 50])
    def test_random_sums_preserved(self, n):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = rng.random(n) * 10
            assert utils.lrRound(v).sum() == round(v.sum())

    def test_row_and_col_round(self):
        m = np.full((3, 4), 1.25)
        assert (utils.rowRound(m).sum(axis=1) == 5).all()
        assert (utils.colRound(m).sum(axis=0) == 4).all()


class TestRanges:
    def test_contiguous_one_based_ranges(self):
        assert utils.ranges([3, 2, 4]) == [(1, 3), (4, 5), (6, 9)]

    def test_zero_length_group(self):
        # An empty group yields start > stop, which callers read as "no members"
        assert utils.ranges([2, 0, 1]) == [(1, 2), (3, 2), (3, 3)]


class TestDrawCounts:
    """drawCounts samples without replacement and decrements its input."""

    def test_draws_requested_number(self):
        rng = np.random.default_rng(0)
        v = np.array([5, 5, 5], dtype=np.int64)
        assert len(utils.drawCounts(v, 7, rng)) == 7
        assert v.sum() == 8

    def test_caps_at_available(self):
        rng = np.random.default_rng(0)
        v = np.array([2, 1], dtype=np.int64)
        assert len(utils.drawCounts(v, 99, rng)) == 3
        assert v.sum() == 0

    def test_never_oversamples_a_bin(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            v = rng.integers(0, 6, size=10).astype(np.int64)
            before = v.copy()
            drawn = utils.drawCounts(v, int(rng.integers(0, 30)), rng)
            counts = np.bincount(drawn, minlength=10) if drawn else np.zeros(10, int)
            assert (counts <= before).all()
            assert np.array_equal(before - counts, v)

    def test_empty(self):
        rng = np.random.default_rng(0)
        assert utils.drawCounts(np.zeros(3, dtype=np.int64), 5, rng) == []
        assert utils.drawCounts(np.array([1, 2], dtype=np.int64), 0, rng) == []

    def test_reproducible_with_same_seed(self):
        v1, v2 = np.array([4, 4, 4], np.int64), np.array([4, 4, 4], np.int64)
        a = utils.drawCounts(v1, 6, np.random.default_rng(7))
        b = utils.drawCounts(v2, 6, np.random.default_rng(7))
        assert a == b


class TestPersonData:
    """Traits are config-driven, not hardcoded."""

    def _person(self, names, values):
        schema = utils.TraitSchema(names)
        return utils.PersonData(hh=(1, 2), sample=3, age=40, working=True,
                                commuter=False, schema=schema,
                                trait_values=schema.values_from(values))

    def test_traits_reachable_by_name(self):
        p = self._person(["hispanic", "female"], {"hispanic": True, "female": False})
        assert p.hispanic is True and p.female is False

    def test_missing_trait_value_is_none(self):
        p = self._person(["hispanic", "female"], {"hispanic": True})
        assert p.female is None

    def test_unknown_trait_raises_with_a_useful_message(self):
        p = self._person(["hispanic"], {"hispanic": True})
        with pytest.raises(AttributeError, match="race_asian_alone"):
            _ = p.race_asian_alone

    def test_arbitrary_new_trait_needs_no_code_change(self):
        # The regression behind issue #2: adding a trait used to raise TypeError
        p = self._person(["some_brand_new_trait"], {"some_brand_new_trait": True})
        assert p.some_brand_new_trait is True

    def test_core_fields_are_not_shadowed_by_traits(self):
        p = self._person(["age"], {"age": 999})
        assert p.age == 40  # the real field wins

    def test_traits_property_round_trips(self):
        p = self._person(["a", "b"], {"a": True, "b": False})
        assert p.traits == {"a": True, "b": False}

    def test_no_instance_dict(self):
        p = self._person([], {})
        assert not hasattr(p, "__dict__")


class TestSmallHelpers:
    def test_indexer_assigns_stable_ids(self):
        idx, d = utils.Indexer(), {}
        assert idx(d, "a") == 1
        assert idx(d, "b") == 2
        assert idx(d, "a") == 1

    def test_thresh(self):
        assert utils.thresh(5, 10) == 0
        assert utils.thresh(15, 10) == 15

    def test_vecmerge_concatenates_by_key(self):
        assert utils.vecmerge({"a": [1]}, {"a": [2], "b": [3]}) == {"a": [1, 2], "b": [3]}

    def test_vecmerge_copies_inputs(self):
        a = {"x": [1]}
        utils.vecmerge(a, {})["x"].append(99)
        assert a == {"x": [1]}

    def test_dflat(self):
        assert sorted(utils.dflat({"a": [1, 2]})) == [("a", 1), ("a", 2)]

    def test_tryjson_missing_file_returns_empty(self, tmp_path):
        assert utils.tryJSON(str(tmp_path / "nope.json")) == {}
