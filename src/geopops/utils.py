# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Johns Hopkins University
"""
Data structures and utility functions for GeoPops synthetic population generation.
Translated from julia/utils.jl.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json
import os
import warnings

# Package directory (src/geopops/), where the shipped config.json template lives.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def tryJSON(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def resolve_config(config, data_dir):
    """Use the in-memory config when given; otherwise read data_dir/config.json.

    Pipeline stages take a `config` argument so that the config resolved once at
    the entry point is the one every stage actually uses. The disk path below is
    only reached when a stage is called directly against an already-generated
    population.

    If the data-dir config is missing or unparseable, fall back to the package
    config and say so. Silence here is dangerous: every caller reads values as
    `config.get(key, <hardcoded default>)`, so an empty dict means the run
    quietly uses source-code literals instead of the user's settings, completes
    normally, and produces plausible-looking wrong numbers.
    """
    if config is not None:
        return config

    data_cfg_path = os.path.join(data_dir, 'config.json')
    cfg = tryJSON(data_cfg_path)
    if cfg:
        return cfg

    reason = "is missing" if not os.path.exists(data_cfg_path) else "could not be read"
    pkg_cfg_path = os.path.join(PACKAGE_DIR, 'config.json')
    pkg_cfg = tryJSON(pkg_cfg_path)
    if pkg_cfg:
        warnings.warn(
            f"Config {data_cfg_path!r} {reason}; falling back to the package "
            f"config {pkg_cfg_path!r}. These can describe different populations "
            f"-- pass config= explicitly to control which one is used.",
            stacklevel=2,
        )
        return pkg_cfg

    warnings.warn(
        f"Config {data_cfg_path!r} {reason} and the package config "
        f"{pkg_cfg_path!r} is unavailable; falling back to built-in defaults, "
        f"which probably do not match your run.",
        stacklevel=2,
    )
    return {}


class TraitSchema:
    """Shared name -> position mapping for the per-person traits of one run.

    One instance is shared by every PersonData in a population, so carrying
    config-driven traits costs a tuple of values per person rather than a set of
    named fields. Keeping the trait list config-driven also means adding a trait
    needs no code change -- the previous hardcoded field list meant a trait in
    the config that PersonData didn't declare died with an opaque TypeError.
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

    def __eq__(self, other):
        return isinstance(other, TraitSchema) and self.names == other.names

    def __hash__(self):
        return hash(self.names)

    def __repr__(self):
        return f"TraitSchema({list(self.names)!r})"


EMPTY_SCHEMA = TraitSchema()


@dataclass(slots=True)
class PersonData:
    """One synthetic person.

    The core demographic fields are fixed. Everything in the config's
    ``additional_traits`` (sex, race/ethnicity, school sector, ...) is carried in
    ``trait_values``, positioned by a TraitSchema shared across the whole
    population, and is still reachable by name: ``person.hispanic`` works
    whenever ``hispanic`` was requested for this run.

    Memory matters here -- one instance exists per person, so hundreds of
    thousands per county. ``slots=True`` removes each instance's ``__dict__``.
    """
    hh: tuple
    sample: int
    age: int
    working: bool
    commuter: bool
    com_cat: Optional[int] = None
    com_inc: Optional[int] = None
    sch_grade: Optional[str] = None
    schema: "TraitSchema" = EMPTY_SCHEMA
    trait_values: tuple = ()

    def __getattr__(self, name):
        # Only reached when normal slot lookup fails, so this can never shadow a
        # real field. object.__getattribute__ avoids recursing back through here.
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
        """The person's config-driven traits as a {name: value} dict."""
        return dict(zip(self.schema.names, self.trait_values))


@dataclass
class Household:
    sample: int
    people: list


@dataclass
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
    return list(zip(starts.tolist(), x.tolist()))


def drawCounts(v, n=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    result = []
    n = min(n, int(v.sum()))
    for _ in range(n):
        total = v.sum()
        if total == 0:
            break
        probs = v.astype(float) / total
        idx = rng.choice(len(v), p=probs)
        v[idx] -= 1
        result.append(idx)
    return result


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
