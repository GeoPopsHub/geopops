"""
Workplace generation, worker assignment, and commute matrix IPF.
Translated from julia/workplaces.jl.
"""
import numpy as np
import pandas as pd
import os
from scipy import sparse
from .utils import tryJSON, lrRound, lrRound_matrix, rowRound, drawCounts, vecmerge
from .ipfn import ipfn as IPFN


def split_lognormal(n, mu, sigma, draws, rng):
    """Split n into workplace sizes drawn from lognormal(mu, sigma).
    Uses and updates the 'draws' list of unused sizes.
    """
    sizes = []
    while n > 2:
        # Try to reuse an unused draw
        found = None
        for i, d in enumerate(draws):
            if d <= n:
                found = i
                break
        if found is not None:
            sz = draws.pop(found)
        else:
            sz = int(np.ceil(np.exp(mu + sigma * rng.standard_normal())))
            while sz > n:
                draws.append(sz)
                sz = int(np.ceil(np.exp(mu + sigma * rng.standard_normal())))
        sizes.append(sz)
        n -= sz
    if n > 0:
        sizes.append(n)
    return sizes


def group_commuters_by_origin(people, cbgs, ind_codes, rng):
    """Group commuters by industry category and origin CBG.
    Returns dict[cat_code -> dict[cbg_code -> list[(p_id, hh_id, cbg_id, income_cat)]]].
    """
    cbgs_inv = {v: k for k, v in cbgs.items()}

    # Build dataframe of commuters
    rows = []
    for k, v in people.items():
        if v.commuter:
            rows.append({'id': k[0], 'hh': k[1], 'cbg': k[2],
                         'income': v.com_inc, 'category': v.com_cat})
    if not rows:
        return {cat: {cbg_code: [] for cbg_code in cbgs.keys()} for cat in ind_codes}

    df = pd.DataFrame(rows)
    worker_keys = {cat: {cbg_code: [] for cbg_code in cbgs.keys()} for cat in ind_codes}

    for (cat_idx, cbg_idx), group in df.groupby(['category', 'cbg']):
        if cat_idx is None or pd.isna(cat_idx):
            continue
        cat_code = ind_codes[int(cat_idx) - 1]
        cbg_code = cbgs_inv.get(cbg_idx, '')
        workers = [(int(r['id']), int(r['hh']), int(r['cbg']), r['income'])
                   for _, r in group.iterrows()]
        rng.shuffle(workers)
        worker_keys[cat_code][cbg_code] = workers

    return worker_keys


class DummyGenerator:
    """Creates dummy workers for commuters from outside the synth area."""
    def __init__(self):
        self.idx = 0

    def __call__(self, origin, inc_code):
        self.idx += 1
        d_err = 0 if origin == 'outside' else 1
        return (self.idx, 0, 0, int(inc_code)), d_err


def read_workers_by_cat(co_results, data_dir, ind_codes, counties):
    """Read # workers by industry category from HH sample summaries."""
    cat_cols = ['com_ind_' + k for k in ind_codes]
    hh_samps = pd.read_csv(os.path.join(data_dir, 'processed', 'hh_samples.csv'),
                            usecols=['SERIALNO'] + cat_cols, dtype={'SERIALNO': str})
    # Keep HH sample row mapping 1-based for Julia parity.
    hh_idx = dict(zip(hh_samps['SERIALNO'], range(1, len(hh_samps) + 1)))

    workers_by_cat = {k: {} for k in ind_codes}
    for co in counties:
        if co not in co_results:
            continue
        cbg_dict = co_results[co]
        for ori, hhvec in cbg_dict.items():
            for cat_code, cat_col in zip(ind_codes, cat_cols):
                total = sum(
                    hh_samps.iloc[hh_idx[x] - 1][cat_col]
                    for x in hhvec
                    if x in hh_idx and pd.notna(hh_samps.iloc[hh_idx[x] - 1][cat_col])
                )
                workers_by_cat[cat_code][ori] = int(total)
    return workers_by_cat


def read_gq_workers_by_cat(gq_summary, ind_codes):
    """Get # workers in GQs by industry from the GQ summary dataframe."""
    cat_cols = ['ind_' + k for k in ind_codes]
    gq_by_cat = {k: {} for k in ind_codes}
    for _, r in gq_summary.iterrows():
        for cat_code, cat_col in zip(ind_codes, cat_cols):
            gq_by_cat[cat_code][r['geo']] = int(r.get(cat_col, 0))
    return gq_by_cat


def read_od_matrix(data_dir, k, m, n):
    """Read sparse OD proportion matrix for category k."""
    df = pd.read_csv(os.path.join(data_dir, 'processed', f'od_{k}.csv.gz'),
                     dtype={'origin': int, 'dest': int, 'p': float})
    # Julia is 1-based; CSV has 1-based indices
    rows = df['origin'].values - 1
    cols = df['dest'].values - 1
    vals = df['p'].values
    return sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))


def read_outside_origins(data_dir, ind_codes):
    """Counts of workers commuting from outside the synth area."""
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'work_cats_live_outside.csv'))
    tmp = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return {k: int(round(tmp.get('C24030:' + k, 0))) for k in ind_codes}


def calc_od_counts(ind_codes, counties, co_results, gq_summary, data_dir):
    """Calculate origin-destination counts for each industry.
    Returns (origin_labels, dest_labels, od_counts_by_cat).
    od_counts_by_cat: dict[cat_code -> dense numpy array of OD counts]
    """
    origin_df = pd.read_csv(os.path.join(data_dir, 'processed', 'od_rows_origins.csv'))
    dest_df = pd.read_csv(os.path.join(data_dir, 'processed', 'od_columns_dests.csv'))
    origin_labels = origin_df['origin'].astype(str).tolist()
    dest_labels = dest_df['dest'].astype(str).tolist()
    n_rows = len(origin_labels)
    n_cols = len(dest_labels)
    origin_idx = {o: i for i, o in enumerate(origin_labels)}

    print("reading worker counts")
    hhw = read_workers_by_cat(co_results, data_dir, ind_codes, counties)
    gqw = read_gq_workers_by_cat(gq_summary, ind_codes)

    outside = read_outside_origins(data_dir, ind_codes)
    for k in ind_codes:
        hhw[k]['outside'] = outside[k]

    print("calculating origin-destination counts")
    od_counts_by_cat = {}
    for k in ind_codes:
        M = read_od_matrix(data_dir, k, n_rows, n_cols).toarray()
        counts = np.zeros((n_rows, n_cols), dtype=np.int64)
        for code, rownum in origin_idx.items():
            hh_n = hhw[k].get(code, 0) + gqw[k].get(code, 0)
            counts[rownum, :] = lrRound(M[rownum, :] * hh_n)
        od_counts_by_cat[k] = counts

    return origin_labels, dest_labels, od_counts_by_cat


def read_county_stats(data_dir):
    """Employer size stats (lognormal mu, sigma) by county."""
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'work_sizes.csv'),
                     usecols=['county', 'mu_ln', 'sigma_ln'], dtype={'county': str})
    return {r['county']: (r['mu_ln'], r['sigma_ln']) for _, r in df.iterrows()}


def read_school_info(data_dir):
    """Returns (sch_n_teachers, sch_closest_cbg)."""
    distmat = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_sch_distmat.csv'),
                           dtype={'GEOID': str})
    sch_cols = [c for c in distmat.columns if c != 'GEOID']
    closest_cbg = {}
    for s in sch_cols:
        idx = distmat[s].idxmin()
        closest_cbg[s] = distmat.loc[idx, 'GEOID']

    schools = pd.read_csv(os.path.join(data_dir, 'processed', 'schools.csv'),
                          usecols=['NCESSCH', 'TEACHERS'], dtype={'NCESSCH': str})
    sch_n_teachers = dict(zip(schools['NCESSCH'], schools['TEACHERS'].astype(int)))
    return sch_n_teachers, closest_cbg


def read_gq_info(gqs, cbgs, config):
    """Returns (gq_n_emps, gq_cbgs): workers needed and CBG location for each GQ."""
    cbgs_inv = {v: k for k, v in cbgs.items()}
    r_i = float(config.get('inst_res_per_worker', 10))
    r_ni = float(config.get('noninst_res_per_worker', 50))
    e_min = config.get('min_gq_workers', 2)
    ni_types = {'milGQ', 'ninst1864civ'}

    gq_n_emps = {}
    gq_cbgs = {}
    for k, v in gqs.items():
        ratio = r_ni if v.type in ni_types else r_i
        gq_n_emps[k] = max(e_min, int(np.ceil(len(v.residents) / ratio)))
        gq_cbgs[k] = cbgs_inv.get(k[1], '')
    return gq_n_emps, gq_cbgs


def filter_dests(geo, dest_idx, colsums):
    """Find destination indices whose code starts with geo and have workers > 0."""
    return [idx for code, idx in dest_idx.items() if code.startswith(geo) and colsums[idx] > 0]


def pull_inst_workers(count_matrix, dest_idx, origin_labels, workers_by_key, loc_by_key, rng):
    """Pull worker origins for institutions (schools/GQs) from the OD count matrix.
    Modifies count_matrix in place. Returns dict[inst_key -> list[origin_codes]].
    """
    inst_emp_origins = {}
    for inst_id, n_needed in workers_by_key.items():
        cbg = loc_by_key[inst_id]
        geo_areas = [cbg, cbg[:11], cbg[:9], cbg[:7], cbg[:5]]
        colsums = count_matrix.sum(axis=0)
        dest_lists = [filter_dests(geo, dest_idx, colsums) for geo in geo_areas]
        for dl in dest_lists:
            rng.shuffle(dl)
        avail_cols = []
        seen = set()
        for dl in dest_lists:
            for c in dl:
                if c not in seen:
                    avail_cols.append(c)
                    seen.add(c)

        o_idxs = []
        remaining = n_needed
        for col in avail_cols:
            draw_n = min(int(colsums[col]), remaining)
            col_view = count_matrix[:, col].copy()
            drawn = drawCounts(col_view, draw_n, rng)
            count_matrix[:, col] = col_view
            o_idxs.extend(drawn)
            remaining -= draw_n
            if remaining < 1:
                break

        if remaining > 0:
            print(f"  warning: inst {inst_id} short by {remaining} workers")

        inst_emp_origins[inst_id] = [origin_labels[i] for i in o_idxs]
    return inst_emp_origins


def generate_workplaces(count_matrix, dest_idx, origin_labels, county_stats, draws_by_county, cat_idx, rng):
    """Generate workplaces by splitting OD counts into employer-sized chunks.
    Modifies count_matrix in place. Returns dict[WRKkey -> list[origin_codes]].
    """
    work_origins = {}
    for co, draws in draws_by_county.items():
        if co not in county_stats:
            continue
        mu, sigma = county_stats[co]
        sigma = sigma + 0.1  # slight adjustment per Julia code
        dests = {code: idx for code, idx in dest_idx.items() if code[:5] == co}
        for dest_code, col in dests.items():
            n = int(count_matrix[:, col].sum())
            if n > 0:
                sizes = split_lognormal(n, mu, sigma, draws, rng)
                for work_i, emp_size in enumerate(sizes):
                    col_view = count_matrix[:, col].copy()
                    o_idxs = drawCounts(col_view, emp_size, rng)
                    count_matrix[:, col] = col_view
                    work_origins[(work_i + 1, cat_idx, dest_code)] = [origin_labels[i] for i in o_idxs]
    return work_origins


def generate_outside_workplaces(work_outside, cat_idx):
    """Create a separate destination for each person working outside the synth area."""
    result = {}
    i = 0
    for origin, count in work_outside.items():
        for _ in range(int(count)):
            i += 1
            result[(i, cat_idx, 'outside')] = [origin]
    return result


def assign_workers(emp_origins, workers_by_origin, cidx_by_origin, dummy_fn, rng):
    """Assign people to employers based on their commute origins.
    Returns (workers_dict, dummies_list, missing_count, ran_out_dict).
    """
    n_by_origin = {k: len(v) for k, v in workers_by_origin.items()}
    total_workers = sum(n_by_origin.values())
    if total_workers > 0:
        n_high_income = sum(1 for v in workers_by_origin.values() for w in v if w[3] == 2)
        p_high = n_high_income / total_workers
    else:
        p_high = 0.5

    workers = {e_id: [] for e_id in emp_origins}
    dummies = []
    missing_origin = 0
    ran_out = {}

    for e_id, origin_vec in emp_origins.items():
        for origin_key in origin_vec:
            if origin_key in cidx_by_origin:
                cidx_by_origin[origin_key] += 1
                i = cidx_by_origin[origin_key]
                if i <= n_by_origin.get(origin_key, 0):
                    workers[e_id].append(workers_by_origin[origin_key][i - 1])
                else:
                    ran_out[origin_key] = ran_out.get(origin_key, 0) + 1
            else:
                inc_code = 2 if rng.random() < p_high else 1
                dum, d_err = dummy_fn(origin_key, inc_code)
                dummies.append(dum)
                workers[e_id].append(dum)
                missing_origin += d_err

    return workers, dummies, missing_origin, ran_out


def generate_commute_matrices(data_dir):
    """Generate per-industry OD proportion matrices using IPF.
    Writes results to processed/od_*.csv.gz. Called before generate_jobs_and_workers.
    """
    print("generating commute matrices")
    wp_codes = tryJSON(os.path.join(data_dir, 'processed', 'codes.json'))
    ind_codes = wp_codes.get('ind_codes', [])

    io_df = pd.read_csv(os.path.join(data_dir, 'processed', 'work_io_sums.csv'), dtype={'Geo': str})
    total_by_ori = io_df.iloc[:, 1:].values.sum(axis=1, keepdims=True)
    m_ind_ori = rowRound(io_df.iloc[:, 1:].values).T  # (n_ind, n_ori)

    od_df = pd.read_csv(os.path.join(data_dir, 'processed', 'work_od_prop.csv'), dtype={'Geo': str})
    origin_idxs = od_df['Geo'].tolist()
    dest_idxs = list(od_df.columns[1:])
    m_dest_ori_dense = rowRound(total_by_ori * od_df.iloc[:, 1:].values).T  # (n_dest, n_ori)
    m_dest_ori = sparse.csc_matrix(m_dest_ori_dense)

    id_est_df = pd.read_csv(os.path.join(data_dir, 'processed', 'work_id_est_sums.csv'),
                             dtype={0: str})
    m_ind_dest = id_est_df.iloc[:, 1:].values.T  # (n_ind, n_dest)
    col_sums = m_ind_dest.sum(axis=0)
    col_sums[col_sums == 0] = 1
    m_ind_dest_p = m_ind_dest / col_sums

    n_ori = len(origin_idxs)
    n_dest = len(dest_idxs)
    n_ind = len(ind_codes)
    res_iod = [sparse.lil_matrix((n_ori, n_dest), dtype=np.float32) for _ in ind_codes]

    for o in range(n_ori):
        if o % 100 == 0:
            print(f"  {o}/{n_ori}")

        ind_margin = m_ind_ori[:, o].astype(float)
        dest_col = m_dest_ori[:, o].toarray().ravel()
        d_idxs = np.where(dest_col > 0)[0]
        d_margin = dest_col[d_idxs].astype(float)

        if len(d_idxs) == 0:
            if ind_margin.sum() > 0:
                print(f"  warning: no commute data for {origin_idxs[o]}, {int(ind_margin.sum())} workers")
            new_m = np.ones((n_ind, 1))
            # Destination = self or random same-county
            self_idx = None
            for di in range(n_dest):
                if dest_idxs[di] == origin_idxs[o]:
                    self_idx = di
                    break
            if self_idx is None:
                county = origin_idxs[o][:5]
                same_county = [di for di in range(n_dest) if dest_idxs[di][:5] == county]
                self_idx = same_county[0] if same_county else 0
            d_idxs = np.array([self_idx])
        else:
            preserve_od = d_margin[np.newaxis, :] * m_ind_dest_p[:, d_idxs]
            init_m = np.maximum(preserve_od, 0.00001)
            IPF = IPFN(init_m.copy(), [ind_margin, d_margin], [[0], [1]],
                       max_iteration=500, convergence_rate=1e-6)
            new_m = IPF.iteration()
            new_margin = new_m.sum(axis=1, keepdims=True)
            zero_rows = np.isclose(new_margin.ravel(), 0.0)
            safe_margin = new_margin.copy()
            safe_margin[safe_margin == 0] = 1
            new_m = new_m / safe_margin
            if d_margin.sum() > 0:
                new_m[zero_rows, :] = d_margin / d_margin.sum()

        for i in range(n_ind):
            res_iod[i][o, d_idxs] = new_m[i, :]

    # Write results
    print("writing commute matrices")
    proc_dir = os.path.join(data_dir, 'processed')
    pd.DataFrame({'idx': range(1, n_ori + 1), 'origin': origin_idxs}).to_csv(
        os.path.join(proc_dir, 'od_rows_origins.csv'), index=False)
    pd.DataFrame({'idx': range(1, n_dest + 1), 'dest': dest_idxs}).to_csv(
        os.path.join(proc_dir, 'od_columns_dests.csv'), index=False)
    for i, k in enumerate(ind_codes):
        m = res_iod[i].tocsr()
        rows, cols, vals = sparse.find(m)
        df = pd.DataFrame({'origin': rows + 1, 'dest': cols + 1, 'p': vals.astype(np.float32)})
        df.to_csv(os.path.join(proc_dir, f'od_{k}.csv.gz'), index=False, compression='gzip')


def generate_jobs_and_workers(people, cbgs, gqs, co_results, gq_summary, data_dir, random_seed=None):
    """Generate workplaces and assign workers.
    Returns (company_workers, sch_workers, gq_workers, outside_workers, dummies).
    Each *_workers is dict[key -> list[worker_tuple]].
    """
    rng = np.random.default_rng(random_seed)
    config = tryJSON(os.path.join(data_dir, 'config.json'))
    wp_codes = tryJSON(os.path.join(data_dir, 'processed', 'codes.json'))
    ind_codes = wp_codes.get('ind_codes', [])
    ind_idxs = {k: i + 1 for i, k in enumerate(ind_codes)}
    cbgs_inv = {v: k for k, v in cbgs.items()}
    counties = sorted(set(v[:5] for v in cbgs.keys()))

    print("reading commuters in synth pop")
    worker_keys = group_commuters_by_origin(people, cbgs, ind_codes, rng)
    dummy_fn = DummyGenerator()

    # Generate commute matrices
    generate_commute_matrices(data_dir)

    origin_labels, dest_labels, od_counts_by_cat = calc_od_counts(
        ind_codes, counties, co_results, gq_summary, data_dir)
    dest_idx = {d: i for i, d in enumerate(dest_labels)}

    county_stats = read_county_stats(data_dir)
    draws_by_county = {co: [] for co in counties}

    sch_n_emps, sch_cbgs = read_school_info(data_dir)
    gq_n_emps, gq_cbgs = read_gq_info(gqs, cbgs, config)

    all_sch_workers = {}
    all_gq_workers = {}
    all_company_workers = {}
    all_outside_workers = {}
    all_dummies = []

    for ckey in ind_codes:
        print(f"generating workplaces for category {ckey}")

        workers_by_origin = worker_keys[ckey]
        cidx_by_origin = {k: 0 for k in workers_by_origin}

        od_counts = od_counts_by_cat[ckey].copy()
        work_outside_counts = od_counts[:, -1].copy()
        od_counts = od_counts[:, :-1]
        dest_idx_local = {d: i for i, d in enumerate(dest_labels[:-1])}
        work_outside = dict(zip(origin_labels, work_outside_counts))

        # Schools
        if ckey == 'EDU':
            sch_origins = pull_inst_workers(od_counts, dest_idx_local, origin_labels,
                                            sch_n_emps, sch_cbgs, rng)
            sw, sd, sm, sr = assign_workers(sch_origins, workers_by_origin, cidx_by_origin, dummy_fn, rng)
            all_sch_workers[ckey] = sw
            all_dummies.extend(sd)

        # Group quarters
        if ckey == 'ADM_MIL':
            gq_origins = pull_inst_workers(od_counts, dest_idx_local, origin_labels,
                                           gq_n_emps, gq_cbgs, rng)
            gw, gd, gm, gr = assign_workers(gq_origins, workers_by_origin, cidx_by_origin, dummy_fn, rng)
            all_gq_workers[ckey] = gw
            all_dummies.extend(gd)

        # Companies
        work_origins = generate_workplaces(od_counts, dest_idx_local, origin_labels,
                                           county_stats, draws_by_county, ind_idxs[ckey], rng)
        cw, cd, cm, cr = assign_workers(work_origins, workers_by_origin, cidx_by_origin, dummy_fn, rng)
        all_company_workers[ckey] = cw
        all_dummies.extend(cd)

        # Outside workers
        ow, _, _, _ = assign_workers(
            generate_outside_workplaces(work_outside, ind_idxs[ckey]),
            workers_by_origin, cidx_by_origin, dummy_fn, rng)
        all_outside_workers[ckey] = ow

        print(f"  assigned {sum(len(v) for v in cw.values())} to companies")

    # Merge across categories
    company_workers = {}
    for cat_workers in all_company_workers.values():
        company_workers.update(cat_workers)

    sch_workers = {}
    for cat_workers in all_sch_workers.values():
        sch_workers = vecmerge(sch_workers, cat_workers)

    gq_workers_out = {}
    for cat_workers in all_gq_workers.values():
        gq_workers_out = vecmerge(gq_workers_out, cat_workers)

    outside_workers = {}
    for cat_workers in all_outside_workers.values():
        outside_workers.update(cat_workers)

    print(f"\n  total: {sum(len(v) for v in company_workers.values())} company workers, "
          f"{sum(len(v) for v in sch_workers.values())} school workers, "
          f"{sum(len(v) for v in gq_workers_out.values())} GQ workers, "
          f"{sum(len(v) for v in outside_workers.values())} outside workers, "
          f"{len(all_dummies)} dummies")

    return company_workers, sch_workers, gq_workers_out, outside_workers, all_dummies
