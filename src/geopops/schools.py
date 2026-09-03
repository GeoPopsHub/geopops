"""
School assignment logic.
Translated from julia/schools.jl.
"""
import numpy as np
import pandas as pd
import os
from .utils import tryJSON


def read_sch_cap(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'schools.csv'),
                     usecols=['NCESSCH', 'STUDENTS'], dtype={'NCESSCH': str})
    return dict(zip(df['NCESSCH'], df['STUDENTS'], strict=False))


def find_closest(data_dir, n):
    """For each CBG and grade, the `n` nearest schools offering that grade.

    Returns ``{grade_key: {cbg_geoid: [school_id, ...]}}``, nearest first.
    """
    schools = pd.read_csv(os.path.join(data_dir, 'processed', 'schools.csv'),
                          dtype={'NCESSCH': str})
    distmat = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_sch_distmat.csv'),
                           dtype={'GEOID': str})
    if 'G_PK_OFFERED' in schools.columns and 'G_KG_OFFERED' in schools.columns:
        schools['G_PK_OFFERED'] = schools['G_PK_OFFERED'] | schools['G_KG_OFFERED']

    grade_keys = ['p', 'k'] + [str(i) for i in range(1, 13)]
    grade_labels = ['PK', 'KG'] + [str(i) for i in range(1, 13)]
    sch_ids = [c for c in distmat.columns if c != 'GEOID']
    geoids = distmat['GEOID'].to_numpy()

    closest = {}
    for gk, gl in zip(grade_keys, grade_labels, strict=False):
        col = f'G_{gl}_OFFERED'
        if col not in schools.columns:
            continue
        valid_set = set(schools['NCESSCH'].values[schools[col].values.astype(bool)])
        valid_cols = [s for s in sch_ids if s in valid_set]
        if not valid_cols:
            closest[gk] = {geo: [] for geo in geoids}
            continue

        # Vectorized nearest-n. Missing distances become +inf so they sort last and
        # can be filtered out afterwards; argpartition finds the n smallest per row
        # without fully sorting, then only that slice is sorted.
        dists = distmat[valid_cols].to_numpy(dtype=float)
        dists = np.where(np.isnan(dists), np.inf, dists)
        k = min(n, dists.shape[1] - 1) if dists.shape[1] > 1 else 0
        if k > 0:
            part = np.argpartition(dists, k, axis=1)[:, :n]
        else:
            part = np.argsort(dists, axis=1)[:, :n]
        rows = np.arange(dists.shape[0])[:, None]
        order = part[rows, np.argsort(dists[rows, part], axis=1)]

        col_names = np.array(valid_cols, dtype=object)
        closest[gk] = {
            geo: col_names[row_order[np.isfinite(dists[i, row_order])]].tolist()
            for i, (geo, row_order) in enumerate(zip(geoids, order, strict=False))
        }
    return closest


def _get_students_in_school(people, cbgs_inv):
    result = []
    for k, p in people.items():
        if p.sch_grade is not None and p.sch_grade not in ('c', 'g'):
            cbg_code = cbgs_inv.get(k[2], '')
            result.append((k, p.sch_grade, cbg_code))
    result.sort(key=lambda x: (x[0][2], x[0][1]))
    return result


def generate_schools(people, cbgs, data_dir, random_seed=None, config=None):
    rng = np.random.default_rng(random_seed)
    if config is None:
        config = tryJSON(os.path.join(data_dir, 'config.json'))
    n_schools = config.get('n_closest_schools', 4)
    prob_closest = config.get('p_closest_school', 0.9)

    closest = find_closest(data_dir, n_schools)
    cbgs_inv = {v: k for k, v in cbgs.items()}
    p_in_school = _get_students_in_school(people, cbgs_inv)

    sch_capacity = read_sch_cap(data_dir)
    sch_capacity = {k: int(round(v * 0.8)) for k, v in sch_capacity.items()}
    sch_students = {k: [] for k in sch_capacity}

    for pk, gr, geo in p_in_school:
        if gr not in closest or geo not in closest[gr]:
            continue
        opts = closest[gr][geo]
        if not opts:
            continue

        idx_avail = None
        for mult in [1.0, 1.5, 2.5]:
            for i, sch in enumerate(opts):
                cap = sch_capacity.get(sch, 0)
                if mult * cap > len(sch_students.get(sch, [])):
                    idx_avail = i
                    break
            if idx_avail is not None:
                break
        if idx_avail is None:
            idx_avail = 0

        idx_choice = idx_avail if rng.random() < prob_closest else idx_avail + 1
        if idx_choice >= len(opts):
            idx_choice = 0
        chosen = opts[idx_choice]
        if chosen not in sch_students:
            sch_students[chosen] = []
        sch_students[chosen].append(pk)

    return sch_students
