"""
Contact network generation (SBM, small-world, complete) and location matrices.
Translated from julia/netw.jl.
"""
import numpy as np
import networkx as nx
from scipy import sparse
from .utils import tryJSON, lrRound, vecmerge


def connect_SBM(keyvec, K, min_N, assoc_coeff, use_groups=True, rng=None):
    """Connect keys using a stochastic block model network.
    If use_groups, element [3] of each key is the group label.
    Returns list of (source_key, dest_key) edges.
    """
    if rng is None:
        rng = np.random.default_rng()
    keyvec = list(dict.fromkeys(keyvec))  # unique, preserving order
    n = len(keyvec)
    if n < 2:
        return []
    if n < min_N:
        return [(keyvec[i], keyvec[j]) for i in range(n) for j in range(i + 1, n)]

    if use_groups:
        group_labels = list(dict.fromkeys(k[3] for k in keyvec))
        group_indices = [[i for i, k in enumerate(keyvec) if k[3] == g] for g in group_labels]
    else:
        group_indices = [list(range(n))]

    n_vec = [len(gi) for gi in group_indices if len(gi) > 0]
    # Filter out empty groups
    group_indices = [gi for gi in group_indices if len(gi) > 0]
    n_groups = len(n_vec)

    # Build mean-degree matrix (Julia convention)
    w_planted = np.diag(np.full(n_groups, K, dtype=float))
    prop_i = np.array(n_vec, dtype=float) / sum(n_vec)
    w_random = np.tile(prop_i * K, (n_groups, 1))
    c_matrix = assoc_coeff * w_planted + (1 - assoc_coeff) * w_random

    # Cap connections
    n_arr = np.array(n_vec, dtype=float)
    for i in range(n_groups):
        c_matrix[i, :] = np.minimum(c_matrix[i, :], n_arr)
    np.fill_diagonal(c_matrix, np.minimum(np.diag(c_matrix), n_arr - 1))

    # Convert to probability matrix for networkx
    p_matrix = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                denom = max(n_vec[j] - 1, 1)
            else:
                denom = max(n_vec[j], 1)
            p_matrix[i, j] = min(c_matrix[i, j] / denom, 1.0)

    g = nx.stochastic_block_model(n_vec, p_matrix.tolist(), seed=int(rng.integers(2**31)))

    # Map graph indices back to keyvec indices
    keyvec_indices = []
    for gi in group_indices:
        keyvec_indices.extend(gi)

    # Fix 0-degree vertices
    for v in g.nodes():
        if g.degree(v) == 0:
            other = rng.integers(g.number_of_nodes())
            g.add_edge(v, other)

    edges = []
    for u, v in g.edges():
        edges.append((keyvec[keyvec_indices[u]], keyvec[keyvec_indices[v]]))
    return edges


def connect_small_world(keyvec, K, min_N, B, rng=None):
    """Connect keys using Watts-Strogatz small-world network."""
    if rng is None:
        rng = np.random.default_rng()
    keyvec = list(dict.fromkeys(keyvec))
    n = len(keyvec)
    if n < 2:
        return []
    if n < min_N:
        return [(keyvec[i], keyvec[j]) for i in range(n) for j in range(i + 1, n)]
    g = nx.watts_strogatz_graph(n, min(K, n - 1), B, seed=int(rng.integers(2**31)))
    return [(keyvec[u], keyvec[v]) for u, v in g.edges()]


def connect_complete(keyvec):
    """Fully connect all keys."""
    keyvec = list(dict.fromkeys(keyvec))
    n = len(keyvec)
    if n < 2:
        return []
    return [(keyvec[i], keyvec[j]) for i in range(n) for j in range(i + 1, n)]


def sp_from_groups(connect_fn, keygroups, p_idxs):
    """Build sparse adjacency matrix from connected groups.
    connect_fn: function(keyvec) -> list of (src_key, dst_key) edges
    keygroups: list of lists of key tuples (first 3 elements = person key)
    p_idxs: dict[person_key -> integer index]
    Returns scipy sparse matrix.
    """
    src_list = []
    dst_list = []
    for keyvec in keygroups:
        for s_key, d_key in connect_fn(keyvec):
            si = p_idxs.get(s_key[:3])
            di = p_idxs.get(d_key[:3])
            if si is not None and di is not None:
                src_list.append(si)
                dst_list.append(di)

    n = len(p_idxs)
    # Make symmetric
    rows = src_list + dst_list
    cols = dst_list + src_list
    data = [True] * len(rows)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=bool)


def _assign_teachers_to_grades(school_key, students_by_grade, sch_workers_for_school, rng):
    """Assign grade labels to teachers proportional to student counts per grade."""
    teachers = sch_workers_for_school
    if not students_by_grade:
        return [(t[0], t[1], t[2], '0') for t in teachers]
    grades = [s[3] for s in students_by_grade]
    from collections import Counter
    grade_counts = Counter(grades)
    total = len(grades)
    n_teachers = len(teachers)
    grade_list = list(grade_counts.keys())
    proportions = np.array([grade_counts[g] / total for g in grade_list])
    n_per_grade = lrRound(proportions * n_teachers)
    teacher_grades = []
    for g, n in zip(grade_list, n_per_grade):
        teacher_grades.extend([g] * n)
    return [(t[0], t[1], t[2], teacher_grades[i] if i < len(teacher_grades) else '0')
            for i, t in enumerate(teachers)]


def _read_sch_ppl(people, sch_students, sch_workers_raw, rng):
    """Prepare school groups: students with grade + teachers assigned to grades.
    Returns dict[school_code -> list[(p_id, hh_id, cbg_id, grade)]].
    """
    # Add grade to each student key
    sch_students_by_grade = {}
    for sch_code, pkeys in sch_students.items():
        sch_students_by_grade[sch_code] = [(pk[0], pk[1], pk[2], people[pk].sch_grade)
                                            for pk in pkeys if pk in people]

    # Strip income from worker tuples -> person keys
    sch_workers_stripped = {}
    for sch_code, wlist in sch_workers_raw.items():
        sch_workers_stripped[sch_code] = [w[:3] for w in wlist]

    # Assign grades to teachers
    teachers_by_grade = {}
    for sch_code, students in sch_students_by_grade.items():
        workers = sch_workers_stripped.get(sch_code, [])
        teachers_by_grade[sch_code] = _assign_teachers_to_grades(
            sch_code, students, workers, rng)

    return vecmerge(sch_students_by_grade, teachers_by_grade)


def _read_gq_ppl(gqs, gq_workers_raw):
    """Prepare GQ groups: residents + workers (stripped of income tag).
    Returns dict[gq_key -> list[Pkey]].
    """
    gq_res = {k: list(v.residents) for k, v in gqs.items()}
    gq_workers = {k: [w[:3] for w in wlist] for k, wlist in gq_workers_raw.items()}
    return vecmerge(gq_res, gq_workers)


def generate_networks(people, households, gqs, sch_students, company_workers,
                      sch_workers, gq_workers, outside_workers, dummies, config, seed=None):
    """Generate all contact networks.
    Returns (adj_hh, adj_non_hh, adj_wp, adj_sch, adj_gq,
             adj_mat_keys, adj_dummy_keys, adj_out_workers).
    """
    rng = np.random.default_rng(seed)
    work_K = config.get('workplace_K', 8)
    school_K = config.get('school_K', 12)
    gq_K = config.get('gq_K', 12)
    sm_world_B = config.get('netw_B', 0.25)
    work_assoc = config.get('income_associativity_coefficient', 0.9)
    sch_assoc = config.get('school_associativity_coefficient', 0.9)

    # Company workers grouped by employer
    cw_groups = list(company_workers.values())
    # School people (students + teachers) by school
    ppl_in_schools = list(_read_sch_ppl(people, sch_students, sch_workers, rng).values())
    # GQ people (residents + workers)
    ppl_in_gq = list(_read_gq_ppl(gqs, gq_workers).values())
    # Household people
    hh_ppl = {k: v.people for k, v in households.items()}
    ppl_in_hhs = list(hh_ppl.values())

    # Build person index mapping
    real_keys = sorted(people.keys(), key=lambda x: (x[2], x[1], x[0]))
    dummy_keys = [d[:3] for d in dummies]
    n_real = len(real_keys)
    p_idxs = {k: i for i, k in enumerate(real_keys)}
    dummy_idxs = {k: n_real + i for i, k in enumerate(dummy_keys)}

    # adj_mat_keys: ordered person keys for matrix rows/columns
    adj_mat_keys = real_keys + dummy_keys
    # adj_dummy_keys: index -> person key for dummies
    adj_dummy_keys = {n_real + i: k for i, k in enumerate(dummy_keys)}

    # Merge indices for network generation
    all_idxs = dict(p_idxs)
    all_idxs.update(dummy_idxs)

    print("  workplace networks (SBM)")
    adj_wp = sp_from_groups(
        lambda v: connect_SBM(v, work_K, work_K + 2, work_assoc, True, rng),
        cw_groups, all_idxs)

    print("  school networks (SBM)")
    adj_sch = sp_from_groups(
        lambda v: connect_SBM(v, school_K, school_K + 2, sch_assoc, True, rng),
        ppl_in_schools, all_idxs)

    print("  GQ networks (small-world)")
    adj_gq = sp_from_groups(
        lambda v: connect_small_world(v, gq_K, gq_K + 2, sm_world_B, rng),
        ppl_in_gq, all_idxs)

    print("  household networks (complete)")
    adj_hh = sp_from_groups(connect_complete, ppl_in_hhs, all_idxs)

    adj_non_hh = (adj_wp + adj_sch + adj_gq).astype(bool)

    # Outside workers: people with workplace outside synth area
    adj_out_workers = {}
    for wkey, wlist in outside_workers.items():
        for w in wlist:
            pk = w[:3]
            if pk in all_idxs:
                adj_out_workers[all_idxs[pk]] = pk

    return (adj_hh, adj_non_hh, adj_wp, adj_sch, adj_gq,
            adj_mat_keys, adj_dummy_keys, adj_out_workers)


def generate_location_matrices(company_workers, households, cbgs, gqs, adj_mat_keys, p_idxs):
    """Generate location contact matrices for ephemeral contacts.
    Returns (w_loc_mat, res_loc_mat, loc_idxs, w_loc_lookup, res_loc_lookup).
    """
    cbgs_inv = {v: k for k, v in cbgs.items()}
    ni_types = {'milGQ', 'ninst1864civ'}
    n_people = len(adj_mat_keys)

    # Index all people
    p_idx_map = {k: i for i, k in enumerate(adj_mat_keys)}

    # Group by census tract (CBG code minus last character)
    hh_tracts = set()
    for hk, hh in households.items():
        cbg_code = cbgs_inv.get(hk[1], '')
        if cbg_code:
            hh_tracts.add(cbg_code[:-1])

    work_tracts = set()
    for wk in company_workers.keys():
        if len(wk) >= 3 and isinstance(wk[2], str) and wk[2] != 'outside':
            work_tracts.add(wk[2][:-1])

    tracts = sorted(hh_tracts | work_tracts)
    loc_idxs = {t: i for i, t in enumerate(tracts)}
    n_tracts = len(tracts)

    # Workers by tract
    w_rows, w_cols = [], []
    for wk, wlist in company_workers.items():
        if len(wk) < 3 or not isinstance(wk[2], str) or wk[2] == 'outside':
            continue
        tract = wk[2][:-1]
        if tract not in loc_idxs:
            continue
        loc_i = loc_idxs[tract]
        for w in wlist:
            pk = w[:3]
            if pk in p_idx_map:
                w_rows.append(p_idx_map[pk])
                w_cols.append(loc_i)

    w_loc_mat = sparse.csr_matrix(
        (np.ones(len(w_rows), dtype=bool), (w_rows, w_cols)),
        shape=(n_people, n_tracts)) if w_rows else sparse.csr_matrix((n_people, n_tracts), dtype=bool)

    # Residents (HH + non-inst GQ) by tract
    r_rows, r_cols = [], []
    for hk, hh in households.items():
        cbg_code = cbgs_inv.get(hk[1], '')
        if not cbg_code:
            continue
        tract = cbg_code[:-1]
        if tract not in loc_idxs:
            continue
        loc_i = loc_idxs[tract]
        for pk in hh.people:
            if pk in p_idx_map:
                r_rows.append(p_idx_map[pk])
                r_cols.append(loc_i)

    for gk, gq in gqs.items():
        if gq.type not in ni_types:
            continue
        cbg_code = cbgs_inv.get(gk[1], '')
        if not cbg_code:
            continue
        tract = cbg_code[:-1]
        if tract not in loc_idxs:
            continue
        loc_i = loc_idxs[tract]
        for pk in gq.residents:
            if pk in p_idx_map:
                r_rows.append(p_idx_map[pk])
                r_cols.append(loc_i)

    res_loc_mat = sparse.csr_matrix(
        (np.ones(len(r_rows), dtype=bool), (r_rows, r_cols)),
        shape=(n_people, n_tracts)) if r_rows else sparse.csr_matrix((n_people, n_tracts), dtype=bool)

    # Per-person lookups
    w_loc_lookup = {}
    for r, c in zip(w_rows, w_cols):
        w_loc_lookup[r] = c
    res_loc_lookup = {}
    for r, c in zip(r_rows, r_cols):
        res_loc_lookup[r] = c

    return w_loc_mat, res_loc_mat, loc_idxs, w_loc_lookup, res_loc_lookup
