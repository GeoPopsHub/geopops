"""
Data structures and utility functions for GeoPops synthetic population generation.
Translated from julia/utils.jl.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json


def tryJSON(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


@dataclass
class PersonData:
    hh: tuple
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
