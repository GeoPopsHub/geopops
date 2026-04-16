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
    return dict(zip(df['NCESSCH'], df['STUDENTS']))


def find_closest(data_dir, n):
    """For each grade and CBG, find the n closest schools offering that grade."""
    schools = pd.read_csv(os.path.join(data_dir, 'processed', 'schools.csv'),
                          dtype={'NCESSCH': str})
    distmat = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_sch_distmat.csv'),
                           dtype={'GEOID': str})

    # Assume kindergartens also offer preschool
    if 'G_PK_OFFERED' in schools.columns and 'G_KG_OFFERED' in schools.columns:
        schools['G_PK_OFFERED'] = schools['G_PK_OFFERED'] | schools['G_KG_OFFERED']

    grade_keys = ['p', 'k'] + [str(i) for i in range(1, 13)]
    grade_labels = ['PK', 'KG'] + [str(i) for i in range(1, 13)]
    sch_ids = [c for c in distmat.columns if c != 'GEOID']

    closest = {}
    for gk, gl in zip(grade_keys, grade_labels):
        col = f'G_{gl}_OFFERED'
        if col not in schools.columns:
            continue
        mask = schools[col].values.astype(bool)
        valid_schs = schools['NCESSCH'].values[mask]
        valid_set = set(valid_schs)
        valid_cols = [s for s in sch_ids if s in valid_set]

        sch_by_geo = {}
        for _, row in distmat.iterrows():
            geo = row['GEOID']
            dists = [(s, row[s]) for s in valid_cols if pd.notna(row[s])]
            dists.sort(key=lambda x: x[1])
            top = dists[:n]
            sch_by_geo[geo] = [s for s, _ in top]
        closest[gk] = sch_by_geo

    return closest


def _get_students_in_school(people, cbgs_inv):
    """Get list of (person_key, grade, cbg_code) for students (excluding college/grad)."""
    result = []
    for k, p in people.items():
        if p.sch_grade is not None and p.sch_grade not in ('c', 'g'):
            cbg_code = cbgs_inv.get(k[2], '')
            result.append((k, p.sch_grade, cbg_code))
    # Sort by (cbg, household) so kids in same household go to same school
    result.sort(key=lambda x: (x[0][2], x[0][1]))
    return result


def generate_schools(people, cbgs, data_dir, random_seed=None):
    """Assign students to schools. Returns sch_students: dict[school_code -> list[Pkey]]."""
    rng = np.random.default_rng(random_seed)
    config = tryJSON(os.path.join(data_dir, 'config.json'))
    n_schools = config.get('n_closest_schools', 4)
    prob_closest = config.get('p_closest_school', 0.9)

    closest = find_closest(data_dir, n_schools)
    cbgs_inv = {v: k for k, v in cbgs.items()}
    p_in_school = _get_students_in_school(people, cbgs_inv)

    sch_capacity = read_sch_cap(data_dir)
    # Shrink capacities to prevent underfilling non-closest schools
    sch_capacity = {k: int(round(v * 0.8)) for k, v in sch_capacity.items()}

    sch_students = {k: [] for k in sch_capacity}

    for pk, gr, geo in p_in_school:
        if gr not in closest or geo not in closest[gr]:
            continue
        opts = closest[gr][geo]
        if not opts:
            continue

        # Find first school not at capacity
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

        # Choose closest or next closest
        idx_choice = idx_avail if rng.random() < prob_closest else idx_avail + 1
        if idx_choice >= len(opts):
            idx_choice = 0

        chosen = opts[idx_choice]
        if chosen not in sch_students:
            sch_students[chosen] = []
        sch_students[chosen].append(pk)

    print(f"  {sum(len(v) for v in sch_students.values())} students assigned to schools")
    return sch_students
