"""
Data structures and utility functions for GeoPops synthetic population generation.
Translated from julia/utils.jl.
"""
import numpy as np
from dataclasses import dataclass
import json


def tryJSON(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except Exception:
        return {}


class TraitSchema:
    """Shared name -> position mapping for the per-person traits of one run.

    One instance is shared by every :class:`PersonData`, so the per-person cost of
    carrying config-driven traits is a tuple of values rather than a dict of
    name/value pairs.
    """
    __slots__ = ("names", "index")

    def __init__(self, names=()):
        self.names = tuple(names)
        self.index = {name: i for i, name in enumerate(self.names)}

    def values_from(self, mapping):
        """Build a trait-value tuple from a name -> value mapping."""
        return tuple(mapping.get(name) for name in self.names)

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return f"TraitSchema({list(self.names)!r})"


EMPTY_SCHEMA = TraitSchema()


@dataclass(slots=True)
class PersonData:
    """One synthetic person.

    The core demographic fields are fixed. Everything listed in the config's
    ``additional_traits`` (sex, race/ethnicity, school sector, ...) is carried in
    ``trait_values``, positioned by a :class:`TraitSchema` shared across the whole
    population, and is reachable by name --- ``person.hispanic`` works whenever
    ``hispanic`` was requested for this run. Keeping traits config-driven rather
    than hardcoded means adding a trait needs no code change.

    Memory matters here: one instance exists per person. ``slots=True`` plus a
    shared schema costs roughly 240 bytes per person, against ~1.5 kB for a plain
    dataclass with one field per trait.
    """
    hh: tuple
    sample: int
    age: int
    working: bool
    commuter: bool
    com_cat: int | None = None
    com_inc: int | None = None
    sch_grade: str | None = None
    schema: TraitSchema = EMPTY_SCHEMA
    trait_values: tuple = ()

    def __getattr__(self, name):
        # Only reached when normal (slot) lookup fails, so this cannot shadow a
        # real field. Use object.__getattribute__ to avoid recursing on `schema`.
        try:
            schema = object.__getattribute__(self, "schema")
            values = object.__getattribute__(self, "trait_values")
        except AttributeError:
            raise AttributeError(name) from None
        position = schema.index.get(name)
        if position is None or position >= len(values):
            raise AttributeError(
                f"PersonData has no field or trait {name!r}. "
                f"Traits available for this run: {list(schema.names)}"
            )
        return values[position]

    @property
    def traits(self):
        """The person's config-driven traits as a ``{name: value}`` dict."""
        return dict(zip(self.schema.names, self.trait_values, strict=False))


@dataclass(slots=True)
class Household:
    sample: int
    people: list


@dataclass(slots=True)
class GQres:
    type: str
    residents: list


class Indexer:
    def __init__(self):
        self.i = 0

    def __call__(self, d, k):
        if k in d:
            return d[k]
        self.i += 1
        d[k] = self.i
        return self.i


def lrRound(v):
    v = np.asarray(v, dtype=float)
    vrnd = np.floor(v).astype(np.int64)
    verr = v - vrnd
    vrem = int(round(v.sum() - vrnd.sum()))
    if vrem > 0:
        vidxs = np.argsort(verr)[::-1]
        for i in range(vrem):
            vrnd[vidxs[i]] += 1
    return vrnd


def lrRound_matrix(m):
    shape = m.shape
    return lrRound(m.ravel()).reshape(shape)


def rowRound(m):
    res = np.zeros(m.shape, dtype=np.int64)
    for i in range(m.shape[0]):
        res[i, :] = lrRound(m[i, :])
    return res


def colRound(m):
    res = np.zeros(m.shape, dtype=np.int64)
    for j in range(m.shape[1]):
        res[:, j] = lrRound(m[:, j])
    return res


def ranges(vec):
    vec = [int(x) for x in vec]
    x = np.cumsum(vec)
    starts = np.concatenate([[1], x[:-1] + 1]).astype(int)
    return list(zip(starts.tolist(), x.tolist(), strict=False))


def drawCounts(v, n=1, rng=None):
    """Draw `n` items without replacement from a multiset of counts.

    `v` is a vector of counts (e.g. commuters per origin); this draws `n` of them
    without replacement, decrements `v` in place, and returns the drawn positions.

    That is exactly the multivariate hypergeometric distribution, so one vectorized
    call replaces the former loop of `n` `rng.choice(p=...)` calls (each of which
    re-normalized the full probability vector). The result is shuffled so callers
    still see draws in random rather than bin order.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = min(int(n), int(v.sum()))
    if n <= 0:
        return []
    drawn = rng.multivariate_hypergeometric(v, n)
    v -= drawn
    result = np.repeat(np.arange(len(v)), drawn)
    rng.shuffle(result)
    return result.tolist()


def thresh(x, v):
    return 0 if x < v else x


def vecmerge(*dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result:
                result[k] = result[k] + v
            else:
                result[k] = list(v)
    return result


def dflat(d):
    result = []
    for k, vlist in d.items():
        for v in vlist:
            result.append((k, v))
    return result


def first_true(bools):
    for i, b in enumerate(bools):
        if b:
            return i
    return None
