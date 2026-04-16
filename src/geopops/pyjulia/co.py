"""
Combinatorial optimization via simulated annealing.
Translated from julia/CO.jl — parallelism removed, uses numpy.
"""
import numpy as np
import pandas as pd
import os
from .utils import tryJSON


def FTdist(v1, v2):
    """Freeman-Tukey distance (divided by 4). Adds 1 to relax zero-cell penalty."""
    return float(np.sum((np.sqrt(v1 + 1.0) - np.sqrt(v2 + 1.0)) ** 2))


def anneal(all_samples, mask, targ, n, params, rng):
    """Simulated annealing on a subset of samples defined by mask.
    Returns (global_indices, generations, score, temperature).
    """
    samples = all_samples[mask, :]
    n_samples = samples.shape[0]
    if n_samples == 0:
        return (np.array([], dtype=int), 0, float('inf'), 0.0)

    maxgens = params['maxgens']
    critval = params['critval']
    cooldown = params['cooldown']

    c0 = rng.integers(0, n_samples, size=n)
    summary = samples[c0, :].sum(axis=0, keepdims=True)
    E0 = FTdist(summary, targ)
    T = 0.5 * E0
    gen = 0

    while True:
        gen += 1
        # mutate
        cidx = rng.integers(len(c0))
        orig = c0[cidx]
        c0[cidx] = rng.integers(n_samples)
        summary = samples[c0, :].sum(axis=0, keepdims=True)
        E1 = FTdist(summary, targ)

        neg_dE = E0 - E1
        if neg_dE >= 0 or rng.random() < np.exp(neg_dE / max(T, 1e-30)):
            T = T * cooldown
            E0 = E1
        else:
            c0[cidx] = orig

        if E0 < critval or gen > maxgens:
            break

    global_indices = np.where(mask)[0]
    res = global_indices[c0]
    return (res, gen, E0, T)


def read_targets(data_dir):
    acs = pd.read_csv(os.path.join(data_dir, 'processed', 'acs_targets.csv'), dtype={'Geo': str})
    geos = acs['Geo'].tolist()
    targs = acs.iloc[:, 1:].values.astype(np.int64)
    colnames = list(acs.columns[1:])
    return targs, geos, colnames


def read_hh_counts(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'hh_counts.csv'), dtype={'Geo': str})
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def read_samples(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'census_samples.csv'), dtype={'SERIALNO': str})
    hh_ids = df['SERIALNO'].values
    samples = df.iloc[:, 1:].values.astype(np.int64)
    return samples, hh_ids


def read_targ_geo(data_dir):
    cols = ['Geo', 'st_puma', 'cbsa', 'county', 'R', 'U']
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_geo.csv'), usecols=cols, dtype={'Geo': str, 'st_puma': str, 'cbsa': str, 'county': str})
    cbg_puma = dict(zip(df['Geo'], df['st_puma']))
    cbg_county = dict(zip(df['Geo'], df['county']))
    cbg_cbsa = dict(zip(df['Geo'], df['cbsa']))
    cbg_urban = dict(zip(df['Geo'], df['U']))
    return cbg_puma, cbg_county, cbg_cbsa, cbg_urban


def read_samp_geo(data_dir):
    cols = ['SERIALNO', 'st_puma', 'cbsa', 'county', 'R', 'U']
    return pd.read_csv(os.path.join(data_dir, 'processed', 'samp_geo.csv'), usecols=cols,
                        dtype={'SERIALNO': str, 'st_puma': str, 'cbsa': str, 'county': str})


def urbanization_lookup(U_values, x):
    """Boolean mask for samples matching urbanization level x."""
    U = U_values.values if hasattr(U_values, 'values') else np.asarray(U_values)
    U = np.where(np.isnan(U), -1, U)
    if x > 0.999:
        return U > 0.999
    elif x < 0.334:
        return U < 0.334
    else:
        a = x - 0.1
        b = min(x + 0.1, 0.999)
        return (U > a) & (U < b)


def sample_lookup(samp_geo, col, target_vals):
    """For each target value, return a boolean mask over samples."""
    if col == 'U':
        return [urbanization_lookup(samp_geo['U'], x) for x in target_vals]
    else:
        vals = samp_geo[col].values
        nan_mask = pd.isna(vals)
        return [np.where(nan_mask, False, vals == x) for x in target_vals]


def optimize(samples, samp_masks, targs, n_hhs, params, rng):
    """Run annealing for each target. Returns list of (indices, gen, score, temp)."""
    results = []
    for mask, targ_row, n in zip(samp_masks, range(len(targs)), n_hhs):
        targ = targs[targ_row:targ_row+1, :]
        r = anneal(samples, mask, targ, n, params, rng)
        results.append(r)
    return results


def reoptimize(x, rerun, samples, samp_masks, targs, n_hhs, params, rng):
    """Re-run optimization on targets that scored poorly; update x in-place."""
    enough = [samp_masks[i].sum() > (n_hhs[rerun[j]] // 2) for j, i in enumerate(range(len(samp_masks)))]
    valid = [rerun[j] for j, ok in enumerate(enough) if ok]
    valid_mask_idx = [j for j, ok in enumerate(enough) if ok]
    if not valid:
        return
    for j, ri in enumerate(valid):
        mi = valid_mask_idx[j]
        targ = targs[ri:ri+1, :]
        r = anneal(samples, samp_masks[mi], targ, n_hhs[ri], params, rng)
        if r[2] < x[ri][2]:
            x[ri] = r


def process_counties(data_dir, counties=None, random_seed=None):
    """Run CO for all counties. Returns (co_results, co_scores).
    co_results: dict[county_code -> dict[cbg_code -> list[serial_number]]]
    co_scores:  dict[county_code -> dict[cbg_code -> float]]
    """
    rng = np.random.default_rng(random_seed)

    samples, hh_ids = read_samples(data_dir)
    cbg_puma, cbg_county, cbg_cbsa, cbg_urban = read_targ_geo(data_dir)
    samp_geo = read_samp_geo(data_dir)
    targs_all, geos_all, _ = read_targets(data_dir)
    hh_counts = read_hh_counts(data_dir)
    n_hhs_all = [hh_counts[g] for g in geos_all]
    county_of = [g[:5] for g in geos_all]

    config = tryJSON(os.path.join(data_dir, 'config.json'))
    c_val = config.get('CO_crit_val', 10.0)
    CO_cooldown = config.get('CO_cooldown', 0.99)
    CO_cooldown_slow = 0.5 + 0.5 * CO_cooldown
    CO_maxgens = config.get('CO_maxgens', 200000)

    params = dict(maxgens=CO_maxgens, critval=c_val, cooldown=CO_cooldown)

    if counties is None:
        counties = sorted(set(county_of))

    all_co_results = {}
    all_co_scores = {}

    for c in counties:
        cmask = [co == c for co in county_of]
        idxs = [i for i, m in enumerate(cmask) if m]
        geos = [geos_all[i] for i in idxs]
        targs = targs_all[idxs, :]
        n_hhs = [n_hhs_all[i] for i in idxs]

        print(f"\nCounty {c}: {len(geos)} CBGs")

        # PUMA level
        print("  puma")
        samp_masks = sample_lookup(samp_geo, 'st_puma', [cbg_puma[g] for g in geos])
        x = optimize(samples, samp_masks, targs, n_hhs, params, rng)
        scores = [a[2] for a in x]
        n_bad = sum(1 for s in scores if s > c_val)
        print(f"  county {c}: {len(x)} targets; {n_bad} above threshold (will rerun); "
              f"E0 min/mean/max: {min(scores):.2f} / {sum(scores)/len(scores):.2f} / {max(scores):.2f}")

        # County level retry
        print("  county")
        rerun = [i for i, r in enumerate(x) if r[2] > c_val]
        print(f"  county pass: {len(rerun)} targets to rerun")
        if rerun:
            re_masks = sample_lookup(samp_geo, 'county', [cbg_county[geos[i]] for i in rerun])
            reoptimize(x, rerun, samples, re_masks, targs, n_hhs, params, rng)
        scores = [a[2] for a in x]
        n_bad = sum(1 for s in scores if s > c_val)
        print(f"  after county: {n_bad} still above threshold; "
              f"E0 min/mean/max: {min(scores):.2f} / {sum(scores)/len(scores):.2f} / {max(scores):.2f}")

        # CBSA level retry
        print("  cbsa")
        rerun = [i for i, r in enumerate(x) if r[2] > c_val]
        print(f"  cbsa pass: {len(rerun)} targets to rerun")
        if rerun:
            re_masks = sample_lookup(samp_geo, 'cbsa', [cbg_cbsa[geos[i]] for i in rerun])
            reoptimize(x, rerun, samples, re_masks, targs, n_hhs, params, rng)
        scores = [a[2] for a in x]
        n_bad = sum(1 for s in scores if s > c_val)
        print(f"  after cbsa: {n_bad} still above threshold; "
              f"E0 min/mean/max: {min(scores):.2f} / {sum(scores)/len(scores):.2f} / {max(scores):.2f}")

        # Urbanization level retry (slower cooldown)
        print("  urbanization")
        rerun = [i for i, r in enumerate(x) if r[2] > c_val]
        print(f"  urb pass: {len(rerun)} targets to rerun")
        if rerun:
            params_slow = dict(maxgens=CO_maxgens, critval=c_val, cooldown=CO_cooldown_slow)
            re_masks = sample_lookup(samp_geo, 'U', [cbg_urban[geos[i]] for i in rerun])
            reoptimize(x, rerun, samples, re_masks, targs, n_hhs, params_slow, rng)
        scores = [a[2] for a in x]
        n_bad = sum(1 for s in scores if s > c_val)
        print(f"  after urb: {n_bad} still above threshold; "
              f"E0 min/mean/max: {min(scores):.2f} / {sum(scores)/len(scores):.2f} / {max(scores):.2f}")

        # Store results: cbg_code -> list of household serial numbers
        co_results_county = {}
        co_scores_county = {}
        for i, geo in enumerate(geos):
            indices = x[i][0]
            co_results_county[geo] = hh_ids[indices].tolist() if len(indices) > 0 else []
            co_scores_county[geo] = x[i][2]

        all_co_results[c] = co_results_county
        all_co_scores[c] = co_scores_county

        n_good = sum(1 for s in co_scores_county.values() if s <= c_val)
        print(f"  {n_good}/{len(geos)} CBGs met criterion ({c_val})")

    return all_co_results, all_co_scores
