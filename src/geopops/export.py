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


def _log_export(verbose, msg=""):
    if verbose:
        print(msg)


def _mcon(val):
    if val is None:
        return ""
    if isinstance(val, bool):
        return int(val)
    return val


def export_synthpop(data_dir, cbgs, households, people, sch_students, sch_workers,
                    gqs, gq_workers, company_workers, outside_workers, verbose=True):
    export_dir = os.path.join(data_dir, 'pop_export')
    rel = 'pop_export'
    os.makedirs(export_dir, exist_ok=True)

    _log_export(verbose, "Exporting cbg_id to cbg_geocode crosswalk")
    cbgs_inv = {v: k for k, v in cbgs.items()}
    rows = sorted([(int(idx), code) for idx, code in cbgs_inv.items()])
    pd.DataFrame(rows, columns=['cbg_id', 'cbg_geocode']).to_csv(
        os.path.join(export_dir, 'cbg_idxs.csv'), index=False)
    _log_export(verbose, f"-- {rel}/cbg_idxs.csv")

    _log_export(verbose, "")
    _log_export(verbose, "Exporting people and households")
    hh_rows = sorted([
        (int(k[0]), int(k[1]), int(v.sample), len(v.people))
        for k, v in households.items()
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(hh_rows, columns=['hh_id', 'cbg_id', 'sample_index', 'n_people']).to_csv(
        os.path.join(export_dir, 'hh.csv'), index=False)
    _log_export(verbose, f"-- {rel}/hh.csv")

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
    _log_export(verbose, f"-- {rel}/people.csv")

    _log_export(verbose, "")
    _log_export(verbose, "Exporting school data")
    sch_flat = dflat(sch_students)
    s_rows = sorted([
        (str(sch), int(pk[0]), int(pk[1]), int(pk[2]))
        for sch, pk in sch_flat
    ], key=lambda x: x[0])
    pd.DataFrame(s_rows, columns=['sch_code', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'sch_students.csv'), index=False)
    _log_export(verbose, f"-- {rel}/sch_students.csv")

    sw_flat = dflat(sch_workers)
    sw_rows = sorted([
        (str(sch), int(w[0]), int(w[1]), int(w[2]))
        for sch, w in sw_flat
    ], key=lambda x: x[0])
    pd.DataFrame(sw_rows, columns=['sch_code', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'sch_workers.csv'), index=False)
    _log_export(verbose, f"-- {rel}/sch_workers.csv")

    _log_export(verbose, "")
    _log_export(verbose, "Exporting group quarters data")
    gq_rows = sorted([
        (int(k[0]), int(k[1]), str(v.type), len(v.residents))
        for k, v in gqs.items()
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(gq_rows, columns=['gq_id', 'cbg_id', 'gq_type', 'n_residents']).to_csv(
        os.path.join(export_dir, 'gqs.csv'), index=False)
    _log_export(verbose, f"-- {rel}/gqs.csv")

    gqr_flat = [(k, pk) for k, v in gqs.items() for pk in v.residents]
    gqr_rows = sorted([
        (int(k[0]), int(k[1]), int(pk[0]), int(pk[1]))
        for k, pk in gqr_flat
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(gqr_rows, columns=['gq_id', 'cbg_id', 'p_id', 'hh_id']).to_csv(
        os.path.join(export_dir, 'gq_residents.csv'), index=False)
    _log_export(verbose, f"-- {rel}/gq_residents.csv")

    gqw_flat = dflat(gq_workers)
    gqw_rows = sorted([
        (int(k[0]), int(k[1]), int(w[0]), int(w[1]), int(w[2]))
        for k, w in gqw_flat
    ], key=lambda x: (x[1], x[0]))
    pd.DataFrame(gqw_rows, columns=['gq_id', 'gq_cbg_id', 'p_id', 'p_hh_id', 'p_cbg_id']).to_csv(
        os.path.join(export_dir, 'gq_workers.csv'), index=False)
    _log_export(verbose, f"-- {rel}/gq_workers.csv")

    _log_export(verbose, "")
    _log_export(verbose, "Exporting workplace data")
    cw_flat = dflat(company_workers)
    cw_rows = sorted([
        (str(k[2]), int(k[1]), int(k[0]), int(w[0]), int(w[1]), int(w[2]))
        for k, w in cw_flat
    ], key=lambda x: (x[0], x[1], x[2]))
    pd.DataFrame(cw_rows, columns=[
        'employer_geo_code', 'employer_type', 'employer_num', 'p_id', 'p_hh_id', 'p_cbg_id'
    ]).to_csv(os.path.join(export_dir, 'company_workers.csv'), index=False)
    _log_export(verbose, f"-- {rel}/company_workers.csv")

    ow_flat = dflat(outside_workers)
    ow_rows = sorted([
        (int(w[0]), int(w[1]), int(w[2]))
        for _, w in ow_flat
    ], key=lambda x: (x[2], x[1], x[0]))
    pd.DataFrame(ow_rows, columns=['p_id', 'p_hh_id', 'p_cbg_id']).to_csv(
        os.path.join(export_dir, 'outside_workers.csv'), index=False)
    _log_export(verbose, f"-- {rel}/outside_workers.csv")


def export_networks(data_dir, adj_hh, adj_non_hh, adj_wp, adj_sch, adj_gq,
                    adj_mat_keys, adj_dummy_keys, adj_out_workers, verbose=True):
    export_dir = os.path.join(data_dir, 'pop_export')
    rel = 'pop_export'
    os.makedirs(export_dir, exist_ok=True)

    def write_upper(name, mat):
        ut = sparse.triu(mat, format='coo')
        ut_int = sparse.coo_matrix((np.ones(ut.nnz, dtype=int), (ut.row, ut.col)), shape=ut.shape)
        mmwrite(os.path.join(export_dir, f'adj_upper_triang_{name}.mtx'), ut_int)

    _log_export(verbose, "")
    _log_export(verbose, "Exporting network data")
    write_upper('hh', adj_hh)
    _log_export(verbose, f"-- {rel}/adj_upper_triang_hh.mtx")
    write_upper('non_hh', adj_non_hh)
    _log_export(verbose, f"-- {rel}/adj_upper_triang_non_hh.mtx")
    write_upper('wp', adj_wp)
    _log_export(verbose, f"-- {rel}/adj_upper_triang_wp.mtx")
    write_upper('sch', adj_sch)
    _log_export(verbose, f"-- {rel}/adj_upper_triang_sch.mtx")
    write_upper('gq', adj_gq)
    _log_export(verbose, f"-- {rel}/adj_upper_triang_gq.mtx")

    key_rows = [(i + 1, i, int(k[0]), int(k[1]), int(k[2])) for i, k in enumerate(adj_mat_keys)]
    pd.DataFrame(key_rows, columns=['index_one', 'index_zero', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'adj_mat_keys.csv'), index=False)
    _log_export(verbose, f"-- {rel}/adj_mat_keys.csv")

    dk_rows = sorted([
        (int(idx) + 1, int(idx), int(pk[0]), int(pk[1]), int(pk[2]))
        for idx, pk in adj_dummy_keys.items()
    ], key=lambda x: x[0])
    pd.DataFrame(dk_rows, columns=['index_one', 'index_zero', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'adj_dummy_keys.csv'), index=False)
    _log_export(verbose, f"-- {rel}/adj_dummy_keys.csv")

    ow_rows = sorted([
        (int(idx) + 1, int(idx), int(pk[0]), int(pk[1]), int(pk[2]))
        for idx, pk in adj_out_workers.items()
    ], key=lambda x: x[0])
    pd.DataFrame(ow_rows, columns=['index_one', 'index_zero', 'p_id', 'hh_id', 'cbg_id']).to_csv(
        os.path.join(export_dir, 'adj_out_workers.csv'), index=False)
    _log_export(verbose, f"-- {rel}/adj_out_workers.csv")
