"""
Data structures and utility functions for GeoPops synthetic population generation.
Translated from julia/utils.jl.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json
import os


def tryJSON(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


@dataclass
class PersonData:
    hh: tuple           # (hh_num, cbg_index)
    sample: int
    age: int
    working: bool
    commuter: bool
    com_cat: Optional[int] = None
    com_inc: Optional[int] = None
    sch_grade: Optional[str] = None
    sch_public: Optional[bool] = None
    sch_private: Optional[bool] = None
    female: Optional[bool] = None
    race_black_alone: Optional[bool] = None
    white_non_hispanic: Optional[bool] = None
    hispanic: Optional[bool] = None


@dataclass
class Household:
    sample: int
    people: list        # list of Pkey = (p_num, hh_num, cbg_key)


@dataclass
class GQres:
    type: str
    residents: list     # list of Pkey


class Indexer:
    """Auto-assigns sequential 1-based integer IDs to new dict keys."""
    def __init__(self):
        self.i = 0

    def __call__(self, d, k):
        if k in d:
            return d[k]
        self.i += 1
        d[k] = self.i
        return self.i


def lrRound(v):
    """Round a 1-D array to integers preserving sum (largest-remainder method)."""
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
    """Round a matrix to integers preserving total sum."""
    shape = m.shape
    return lrRound(m.ravel()).reshape(shape)


def rowRound(m):
    """Round each row of a matrix preserving row sums."""
    res = np.zeros(m.shape, dtype=np.int64)
    for i in range(m.shape[0]):
        res[i, :] = lrRound(m[i, :])
    return res


def colRound(m):
    """Round each column of a matrix preserving column sums."""
    res = np.zeros(m.shape, dtype=np.int64)
    for j in range(m.shape[1]):
        res[:, j] = lrRound(m[:, j])
    return res


def ranges(vec):
    """Continuous 1-based index ranges with lengths given by vec.
    Returns list of (start, end) tuples (1-based, inclusive).
    """
    vec = [int(x) for x in vec]
    x = np.cumsum(vec)
    starts = np.concatenate([[1], x[:-1] + 1]).astype(int)
    return list(zip(starts.tolist(), x.tolist()))


def drawCounts(v, n=1, rng=None):
    """Sample n indices from a vector of counts (weighted), depleting counts in-place.
    v is modified in place. Returns list of drawn indices.
    """
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
    """Replace x with 0 if x < v."""
    return 0 if x < v else x


def vecmerge(*dicts):
    """Merge dicts whose values are lists, concatenating values for shared keys."""
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result:
                result[k] = result[k] + v
            else:
                result[k] = list(v)
    return result


def dflat(d):
    """Flatten a dict of {key: [values]} into [(key, value), ...]."""
    result = []
    for k, vlist in d.items():
        for v in vlist:
            result.append((k, v))
    return result


def first_true(bools):
    """Index of first True in a sequence, or None."""
    for i, b in enumerate(bools):
        if b:
            return i
    return None
