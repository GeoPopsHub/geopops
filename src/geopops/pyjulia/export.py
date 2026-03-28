"""
Export synthetic population and networks to CSV/MTX files.
Translated from julia/export_synthpop.jl and julia/export_network.jl.
"""
import numpy as np
import pandas as pd
import os
from scipy import sparse
from scipy.io import mmwrite
from .utils import dflat


def _mcon(val):
    """Convert value for export: None -> empty, bool -> int, else pass through."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return int(val)
    return val


def export_synthpop(data_dir, cbgs, households, people, sch_students, sch_workers,
                    gqs, gq_workers, company_workers, outside_workers):
    """Write population data to CSV files in pop_export/."""
    export_dir = os.path.join(data_dir, 'pop_export')
    os.makedirs(export_dir, exist_ok=True)

    # CBG index mapping
    cbgs_inv = {v: k for k, v in cbgs.items()}
    rows = sorted([(int(idx), code) for idx, code in cbgs_inv.items()])
    pd.DataFrame(rows, columns=['cbg_id', 'cbg_geocode']).to_csv(
        os.path.join(export_dir, 'cbg_idxs.csv'), index=False)

    # Households
    hh_rows = sorted([
        (int(k[0]), int(k[1]), int(v.sample), len(v.people))
        for k, v in households.items()
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(hh_rows, columns=['hh_id', 'cbg_id', 'sample_index', 'n_people']).to_csv(
        os.path.join(export_dir, 'hh.csv'), index=False)

    # People
    p_rows = sorted([
        (int(k[0]), int(k[1]), int(k[2]), _mcon(v.sample), _mcon(v.age),
         _mcon(v.female), _mcon(v.working), _mcon(v.commuter),
         _mcon(v.com_inc), _mcon(v.com_cat),
         _mcon(v.race_black_alone), _mcon(v.white_non_hispanic), _mcon(v.hispanic),
         _mcon(v.sch_grade))
        for k, v in people.items()
    ], key=lambda x: (x[2], x[1], x[0]))
    pd.DataFrame(p_rows, columns=[
        'p_id', 'hh_id', 'cbg_id', 'sample_index', 'age', 'female', 'working', 'commuter',
        'commuter_income_category', 'commuter_workplace_category',
        'race_black_alone', 'white_non_hispanic', 'hispanic', 'sch_grade'
    ]).to_csv(os.path.join(export_dir, 'people.csv'), index=False)

    # School students
    sch_flat = dflat(sch_students)
    s_rows = sorted([
        (str(sch), int(pk[0]), int(pk[1]), int(pk[2]))
        for sch, pk in sch_flat
    ], key=lambda x: x[0])
    pd.DataFrame(s_rows, columns=['sch_code', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'sch_students.csv'), index=False)

    # School workers (strip income tag from worker tuples)
    sw_flat = dflat(sch_workers)
    sw_rows = sorted([
        (str(sch), int(w[0]), int(w[1]), int(w[2]))
        for sch, w in sw_flat
    ], key=lambda x: x[0])
    pd.DataFrame(sw_rows, columns=['sch_code', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'sch_workers.csv'), index=False)

    # Group quarters
    gq_rows = sorted([
        (int(k[0]), int(k[1]), str(v.type), len(v.residents))
        for k, v in gqs.items()
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(gq_rows, columns=['gq_id', 'cbg_id', 'gq_type', 'n_residents']).to_csv(
        os.path.join(export_dir, 'gqs.csv'), index=False)

    # GQ residents
    gqr_flat = [(k, pk) for k, v in gqs.items() for pk in v.residents]
    gqr_rows = sorted([
        (int(k[0]), int(k[1]), int(pk[0]), int(pk[1]))
        for k, pk in gqr_flat
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(gqr_rows, columns=['gq_id', 'cbg_id', 'p_id', 'hh_id']).to_csv(
        os.path.join(export_dir, 'gq_residents.csv'), index=False)

    # GQ workers (strip income tag)
    gqw_flat = dflat(gq_workers)
    gqw_rows = sorted([
        (int(k[0]), int(k[1]), int(w[0]), int(w[1]), int(w[2]))
        for k, w in gqw_flat
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(gqw_rows, columns=['gq_id', 'gq_cbg_id', 'p_id', 'p_hh_id', 'p_cbg_id']).to_csv(
        os.path.join(export_dir, 'gq_workers.csv'), index=False)

    # Company workers
    cw_flat = dflat(company_workers)
    cw_rows = sorted([
        (str(k[2]), int(k[1]), int(k[0]), int(w[0]), int(w[1]), int(w[2]))
        for k, w in cw_flat
    ], key=lambda x: (x[0], x[1], x[2]))
    pd.DataFrame(cw_rows, columns=[
        'employer_geo_code', 'employer_type', 'employer_num', 'p_id', 'p_hh_id', 'p_cbg_id'
    ]).to_csv(os.path.join(export_dir, 'company_workers.csv'), index=False)

    # Outside workers
    ow_flat = dflat(outside_workers)
    ow_rows = sorted([
        (int(w[0]), int(w[1]), int(w[2]))
        for _, w in ow_flat
    ], key=lambda x: (x[2], x[1], x[0]))
    pd.DataFrame(ow_rows, columns=['p_id', 'p_hh_id', 'p_cbg_id']).to_csv(
        os.path.join(export_dir, 'outside_workers.csv'), index=False)

    print(f"exported population data to {export_dir}")


def export_networks(data_dir, adj_hh, adj_non_hh, adj_wp, adj_sch, adj_gq,
                    adj_mat_keys, adj_dummy_keys, adj_out_workers):
    """Write adjacency matrices (MTX) and key files (CSV) to pop_export/."""
    export_dir = os.path.join(data_dir, 'pop_export')
    os.makedirs(export_dir, exist_ok=True)

    def write_upper(name, mat):
        """Write upper triangle of sparse bool matrix as MTX."""
        ut = sparse.triu(mat, format='coo')
        # Convert to integer for Matrix Market format
        ut_int = sparse.coo_matrix((np.ones(ut.nnz, dtype=int), (ut.row, ut.col)), shape=ut.shape)
        mmwrite(os.path.join(export_dir, f'adj_upper_triang_{name}.mtx'), ut_int)

    print("  household adjacency matrix")
    write_upper('hh', adj_hh)
    print("  non-household adjacency matrix")
    write_upper('non_hh', adj_non_hh)
    print("  workplace adjacency matrix")
    write_upper('wp', adj_wp)
    print("  school adjacency matrix")
    write_upper('sch', adj_sch)
    print("  group quarters adjacency matrix")
    write_upper('gq', adj_gq)

    # Matrix keys (1-based index)
    key_rows = [(i + 1, i, int(k[0]), int(k[1]), int(k[2])) for i, k in enumerate(adj_mat_keys)]
    pd.DataFrame(key_rows, columns=['index_one', 'index_zero', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'adj_mat_keys.csv'), index=False)

    # Dummy keys (commuters from outside)
    dk_rows = sorted([
        (int(idx) + 1, int(idx), int(pk[0]), int(pk[1]), int(pk[2]))
        for idx, pk in adj_dummy_keys.items()
    ], key=lambda x: x[0])
    pd.DataFrame(dk_rows, columns=['index_one', 'index_zero', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'adj_dummy_keys.csv'), index=False)

    # Outside workers
    ow_rows = sorted([
        (int(idx) + 1, int(idx), int(pk[0]), int(pk[1]), int(pk[2]))
        for idx, pk in adj_out_workers.items()
    ], key=lambda x: x[0])
    pd.DataFrame(ow_rows, columns=['index_one', 'index_zero', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'adj_out_workers.csv'), index=False)

    print(f"exported networks to {export_dir}")
