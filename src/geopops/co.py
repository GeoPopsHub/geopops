"""
Combinatorial optimization via simulated annealing.
Translated from julia/CO.jl — uses numpy.

Each Census Block Group (CBG) is fitted independently: a set of ``n`` PUMS
household samples is chosen so that its aggregate demographic profile matches
the ACS target profile for that CBG, minimizing the Freeman-Tukey distance.

Fitting proceeds in four passes over progressively broader sample pools --- PUMA,
county, CBSA, then urbanization level --- with each pass re-running only the CBGs
that did not yet meet the stopping criterion.
"""
import numpy as np
import pandas as pd
import os
from .utils import tryJSON


def FTdist(v1, v2):
    """Freeman-Tukey distance (divided by 4). Adds 1 to relax zero-cell penalty."""
    return float(np.sum((np.sqrt(v1 + 1.0) - np.sqrt(v2 + 1.0)) ** 2))


def anneal(samples, global_idxs, targ, n, params, rng):
    """Simulated annealing over a pool of candidate household samples.

    Args:
        samples: ``(n_pool, n_cols)`` int array of candidate sample profiles.
        global_idxs: ``(n_pool,)`` int array mapping pool rows back to the full
            sample table.
        targ: ``(1, n_cols)`` target profile for this CBG.
        n: number of households to select.
        params: dict with ``maxgens``, ``critval``, ``cooldown``.
        rng: numpy Generator.

    Returns:
        tuple: ``(global_indices, generations, score, temperature)``.
    """
    n_samples = samples.shape[0]
    if n_samples == 0:
        return (np.array([], dtype=int), 0, float('inf'), 0.0)

    maxgens = params['maxgens']
    critval = params['critval']
    cooldown = params['cooldown']

    # sqrt(targ + 1) is constant across the whole run; hoist it out of the loop.
    sqrt_targ = np.sqrt(np.asarray(targ, dtype=float).ravel() + 1.0)

    c0 = rng.integers(0, n_samples, size=n)
    # `summary` is the column-wise sum of the currently selected samples. Only one
    # selection changes per generation, so it is updated incrementally rather than
    # recomputed --- the inner loop becomes O(n_cols) instead of O(n * n_cols).
    # `samples` is integer, so the incremental update is exact.
    summary = samples[c0, :].sum(axis=0)
    E0 = float(np.sum((np.sqrt(summary + 1.0) - sqrt_targ) ** 2))
    T = 0.5 * E0
    gen = 0

    while True:
        gen += 1
        cidx = rng.integers(len(c0))
        orig = c0[cidx]
        new = rng.integers(n_samples)
        c0[cidx] = new
        summary += samples[new]
        summary -= samples[orig]
        E1 = float(np.sum((np.sqrt(summary + 1.0) - sqrt_targ) ** 2))

        neg_dE = E0 - E1
        if neg_dE >= 0 or rng.random() < np.exp(neg_dE / max(T, 1e-30)):
            T = T * cooldown
            E0 = E1
        else:
            c0[cidx] = orig
            summary += samples[orig]
            summary -= samples[new]

        if E0 < critval or gen > maxgens:
            break

    return (global_idxs[c0], gen, E0, T)


def read_targets(data_dir):
    acs = pd.read_csv(os.path.join(data_dir, 'processed', 'acs_targets.csv'), dtype={'Geo': str})
    geos = acs['Geo'].tolist()
    targs = acs.iloc[:, 1:].values.astype(np.int64)
    colnames = list(acs.columns[1:])
    return targs, geos, colnames


def read_hh_counts(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'hh_counts.csv'), dtype={'Geo': str})
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1], strict=False))


def read_samples(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'census_samples.csv'), dtype={'SERIALNO': str})
    hh_ids = df['SERIALNO'].values
    samples = df.iloc[:, 1:].values.astype(np.int64)
    return samples, hh_ids


def read_targ_geo(data_dir):
    cols = ['Geo', 'st_puma', 'cbsa', 'county', 'R', 'U']
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_geo.csv'), usecols=cols, dtype={'Geo': str, 'st_puma': str, 'cbsa': str, 'county': str})
    cbg_puma = dict(zip(df['Geo'], df['st_puma'], strict=False))
    cbg_county = dict(zip(df['Geo'], df['county'], strict=False))
    cbg_cbsa = dict(zip(df['Geo'], df['cbsa'], strict=False))
    cbg_urban = dict(zip(df['Geo'], df['U'], strict=False))
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
    """Map each target value to the sample rows that match it.

    Returns a list of ``(key, index_array)`` pairs, one per target value. Targets
    that share a key share the *same* index array object, so the caller can cache
    the extracted sub-matrix by key --- CBGs in one county usually fall into only a
    handful of PUMAs, so this collapses hundreds of gathers into a few.
    """
    if col == 'U':
        def mask_for(x):
            return urbanization_lookup(samp_geo['U'], x)
    else:
        vals = samp_geo[col].values
        nan_mask = pd.isna(vals)

        def mask_for(x):
            return np.where(nan_mask, False, vals == x)

    cache = {}
    out = []
    for x in target_vals:
        idx = cache.get(x)
        if idx is None:
            idx = cache[x] = np.flatnonzero(mask_for(x))
        out.append((x, idx))
    return out


def _subpool(samples, key, idxs, cache):
    """Extract (and memoize) the candidate sample sub-matrix for one lookup key."""
    sub = cache.get(key)
    if sub is None:
        sub = samples[idxs, :]
        cache[key] = sub
    return sub


def optimize(samples, samp_lookups, targs, n_hhs, params, rng):
    """Run annealing for each target. Returns list of (indices, gen, score, temp)."""
    cache = {}
    results = []
    for (key, idxs), targ, n in zip(samp_lookups, targs, n_hhs, strict=False):
        sub = _subpool(samples, key, idxs, cache)
        results.append(anneal(sub, idxs, targ[np.newaxis, :], n, params, rng))
    return results


def reoptimize(x, rerun, samples, samp_lookups, targs, n_hhs, params, rng):
    """Re-run optimization on targets that scored poorly; update `x` in place.

    Targets whose candidate pool is smaller than half the households they need are
    skipped --- the pool is too thin for the fit to improve.
    """
    cache = {}
    for j, ri in enumerate(rerun):
        key, idxs = samp_lookups[j]
        if len(idxs) <= (n_hhs[ri] // 2):
            continue
        sub = _subpool(samples, key, idxs, cache)
        r = anneal(sub, idxs, targs[ri:ri + 1, :], n_hhs[ri], params, rng)
        if r[2] < x[ri][2]:
            x[ri] = r


def _score_report(label, x, c_val, log=print):
    """Print min/mean/max fit score and how many CBGs are still above threshold."""
    scores = [a[2] for a in x]
    n_bad = sum(1 for s in scores if s > c_val)
    log(f"-- After {label} pass: {n_bad} still above threshold; "
        f"E0 min/mean/max: {min(scores):.2f} / {sum(scores) / len(scores):.2f} / {max(scores):.2f}")


def process_counties(data_dir, counties=None, random_seed=None, config=None, verbose=1):
    """Run CO for all counties. Returns (co_results, co_scores)."""
    rng = np.random.default_rng(random_seed)
    log = print if verbose else (lambda *a, **k: None)

    samples, hh_ids = read_samples(data_dir)
    cbg_puma, cbg_county, cbg_cbsa, cbg_urban = read_targ_geo(data_dir)
    samp_geo = read_samp_geo(data_dir)
    targs_all, geos_all, _ = read_targets(data_dir)
    hh_counts = read_hh_counts(data_dir)
    n_hhs_all = [hh_counts[g] for g in geos_all]
    county_of = [g[:5] for g in geos_all]

    if config is None:
        config = tryJSON(os.path.join(data_dir, 'config.json'))
    c_val = config.get('CO_crit_val', 10.0)
    CO_cooldown = config.get('CO_cooldown', 0.99)
    CO_maxgens = config.get('CO_maxgens', 200000)

    params = dict(maxgens=CO_maxgens, critval=c_val, cooldown=CO_cooldown)
    # The final (urbanization) pass cools more slowly, exploring longer, because by
    # then the sample pool is broad and the fit is hard.
    params_slow = dict(maxgens=CO_maxgens, critval=c_val, cooldown=0.5 + 0.5 * CO_cooldown)

    if counties is None:
        counties = sorted(set(county_of))

    all_co_results = {}
    all_co_scores = {}

    for c in counties:
        idxs = [i for i, county in enumerate(county_of) if county == c]
        geos = [geos_all[i] for i in idxs]
        targs = targs_all[idxs, :]
        n_hhs = [n_hhs_all[i] for i in idxs]

        log(f"\nCounty {c}: {len(geos)} CBGs\n")
        log(f"Optimizing {len(geos)} CBGs at PUMA level")
        x = optimize(samples, sample_lookup(samp_geo, 'st_puma', [cbg_puma[g] for g in geos]),
                     targs, n_hhs, params, rng)
        _score_report("PUMA", x, c_val, log)

        # Progressively broaden the candidate pool for CBGs that still fit poorly.
        fallbacks = [("county", 'county', cbg_county, params),
                     ("CBSA", 'cbsa', cbg_cbsa, params),
                     ("urbanization", 'U', cbg_urban, params_slow)]
        for label, col, lookup, level_params in fallbacks:
            rerun = [i for i, r in enumerate(x) if r[2] > c_val]
            if not rerun:
                log(f"All CBGs met the criterion; skipping {label} pass")
                continue
            log(f"Optimizing {len(rerun)} CBG(s) at {label} level")
            samp_lookups = sample_lookup(samp_geo, col, [lookup[geos[i]] for i in rerun])
            reoptimize(x, rerun, samples, samp_lookups, targs, n_hhs, level_params, rng)
            _score_report(label, x, c_val, log)

        co_results_county = {}
        co_scores_county = {}
        for i, geo in enumerate(geos):
            indices = x[i][0]
            co_results_county[geo] = hh_ids[indices].tolist() if len(indices) > 0 else []
            co_scores_county[geo] = x[i][2]

        all_co_results[c] = co_results_county
        all_co_scores[c] = co_scores_county

        n_good = sum(1 for s in co_scores_county.values() if s <= c_val)
        log(f"\n{n_good}/{len(geos)} CBGs met the stopping criterion "
            f"(E0 <= {c_val}) for the Freeman-Tukey distance score.")

    return all_co_results, all_co_scores
