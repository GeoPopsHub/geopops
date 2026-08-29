"""
Household, person, and group quarters generation from PUMS samples.
Translated from julia/households.jl.
"""
import numpy as np
import pandas as pd
import os
from .utils import (PersonData, Household, GQres, Indexer, TraitSchema,
                    tryJSON, thresh, ranges)


def read_counties(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_geo.csv'),
                     usecols=['county'], dtype={'county': str})
    return sorted(df['county'].unique().tolist())


def read_hh_serials(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'hh_samples.csv'),
                     usecols=['SERIALNO'], dtype={'SERIALNO': str})
    return dict(zip(df['SERIALNO'], range(1, len(df) + 1), strict=False))


def read_psamp_df(data_dir, ind_codes, additional_traits):
    nonbool_cols = ['SERIALNO', 'AGEP', 'sch_grade']
    bool_cols = ['commuter', 'has_job', 'com_LODES_low', 'com_LODES_high']
    ind_cols = ['ind_' + k for k in ind_codes]
    all_cols = nonbool_cols + bool_cols + additional_traits + ind_cols
    type_dict = {'SERIALNO': str}
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'p_samples.csv'),
                     usecols=all_cols, dtype=type_dict)
    return df[all_cols]


def people_by_serial(p_samps):
    result = {}
    for idx, serial in enumerate(p_samps['SERIALNO']):
        result.setdefault(serial, []).append(idx + 1)
    return result


def _row_gq_employment(n, jobtype, ind_codes, row):
    if jobtype == 'none' or n < 1:
        return [0] * len(ind_codes)
    prefix = 'civ_ind_' if jobtype == 'civ' else 'mil_ind_'
    return [int(round(row.get(prefix + k, 0))) for k in ind_codes]


def generate_group_quarters(config, cbgs, cbg_indexer, ind_codes, data_dir, rng, schema=None):
    min_gq_residents = config.get('min_gq_residents', 20)
    if schema is None:
        schema = TraitSchema(config.get('additional_traits', []))
    # Group-quarters residents come from ACS aggregates rather than PUMS person
    # records, so none of the PUMS-derived traits are known for them.
    gq_trait_values = (None,) * len(schema)

    gq_cols = ['Geo', 'group quarters:', 'group quarters:under 18', 'group quarters:18 to 64',
               'group quarters:65 and over', 'p_u18_inst', 'p_18_64_inst', 'p_65o_inst',
               'p_18_64_noninst_civil', 'p_18_64_noninst_mil',
               'commuter_p|ninst1864civ', 'work_from_home_p|ninst1864civ',
               'com_LODES_low_p|ninst1864civ', 'com_LODES_high_p|ninst1864civ',
               'commuter_p|milGQ', 'work_from_home_p|milGQ',
               'com_LODES_low_p|milGQ', 'com_LODES_high_p|milGQ']
    df_gq = pd.read_csv(os.path.join(data_dir, 'processed', 'group_quarters.csv'),
                         usecols=gq_cols, dtype={'Geo': str})

    gq_types = ['instu18', 'inst1864', 'ninst1864civ', 'milGQ', 'inst65o']
    assumed_ages = [15, 30, 30, 30, 75]
    job_types = ['none', 'none', 'civ', 'mil', 'none']

    df_gq['pop_instu18'] = (df_gq['group quarters:under 18'] * df_gq['p_u18_inst']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_inst1864'] = (df_gq['group quarters:18 to 64'] * df_gq['p_18_64_inst']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_ninst1864civ'] = (df_gq['group quarters:18 to 64'] * df_gq['p_18_64_noninst_civil']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_milGQ'] = (df_gq['group quarters:18 to 64'] * df_gq['p_18_64_noninst_mil']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_inst65o'] = (df_gq['group quarters:65 and over'] * df_gq['p_65o_inst']).apply(lambda x: thresh(int(round(x)), min_gq_residents))

    df_civil = pd.read_csv(os.path.join(data_dir, 'processed', 'gq_civilian_workers.csv'), dtype={'Geo': str})
    df_civil.columns = [c.replace('C24030:', 'civ_ind_').replace('C24010:', 'civ_occ_') for c in df_civil.columns]
    df_mil = pd.read_csv(os.path.join(data_dir, 'processed', 'gq_military_workers.csv'), dtype={'Geo': str})
    df_mil.columns = [c.replace('C24030:', 'mil_ind_').replace('C24010:', 'mil_occ_') for c in df_mil.columns]

    df_gq['commuter_p|civ_worker'] = df_gq['commuter_p|ninst1864civ'] / (df_gq['commuter_p|ninst1864civ'] + df_gq['work_from_home_p|ninst1864civ'])
    df_gq['commuter_p|mil_worker'] = df_gq['commuter_p|milGQ'] / (df_gq['commuter_p|milGQ'] + df_gq['work_from_home_p|milGQ'])
    df_gq['LODES_high|civ_commuter'] = df_gq['com_LODES_high_p|ninst1864civ'] / (df_gq['com_LODES_high_p|ninst1864civ'] + df_gq['com_LODES_low_p|ninst1864civ'])
    df_gq['LODES_high|mil_commuter'] = df_gq['com_LODES_high_p|milGQ'] / (df_gq['com_LODES_high_p|milGQ'] + df_gq['com_LODES_low_p|milGQ'])
    df_gq = df_gq.merge(df_civil, on='Geo').merge(df_mil, on='Geo')

    gqs = {}
    gq_people = {}

    commuter_p_map = {'instu18': 0.0, 'inst1864': 0.0, 'ninst1864civ': None, 'milGQ': 0.0, 'inst65o': 0.0}
    LODES_high_map = {'instu18': 0.0, 'inst1864': 0.0, 'ninst1864civ': None, 'milGQ': None, 'inst65o': 0.0}

    for _, r in df_gq.iterrows():
        cbg_index = cbg_indexer(cbgs, r['Geo'])
        gq_pops = [r['pop_' + t] for t in gq_types]
        emp_stats = [_row_gq_employment(gq_pops[i], job_types[i], ind_codes, r) for i in range(5)]
        p_idxs = ranges(gq_pops)

        cp = dict(commuter_p_map)
        cp['ninst1864civ'] = r.get('commuter_p|civ_worker', 0.0)
        lh = dict(LODES_high_map)
        lh['ninst1864civ'] = r.get('LODES_high|civ_commuter', 0.0)
        lh['milGQ'] = r.get('LODES_high|mil_commuter', 0.0)

        for t_idx, t_code in enumerate(gq_types):
            if gq_pops[t_idx] > 0:
                pkeys = [(p_i, 0, cbg_index) for p_i in range(p_idxs[t_idx][0], p_idxs[t_idx][1] + 1)]
                gqs[(t_idx + 1, cbg_index)] = GQres(t_code, pkeys)
                emp_cumsum = np.cumsum(emp_stats[t_idx])
                for i, k in enumerate(pkeys):
                    person_i = i + 1
                    emp_cat_matches = np.where(emp_cumsum >= person_i)[0]
                    emp_cat = int(emp_cat_matches[0] + 1) if len(emp_cat_matches) > 0 else None
                    has_job = emp_cat is not None
                    commuter_prob = cp.get(t_code, 0.0)
                    if pd.isna(commuter_prob):
                        commuter_prob = 0.0
                    is_commuter = has_job and (rng.random() < commuter_prob)
                    lh_prob = lh.get(t_code, 0.0)
                    if pd.isna(lh_prob):
                        lh_prob = 0.0
                    inc_cat = (2 if rng.random() < lh_prob else 1) if is_commuter else None
                    emp_cat_final = emp_cat if is_commuter else None
                    gq_people[k] = PersonData(
                        hh=(0, cbg_index), sample=0, age=assumed_ages[t_idx],
                        working=has_job, commuter=is_commuter,
                        com_cat=emp_cat_final, com_inc=inc_cat,
                        sch_grade=None,
                        schema=schema, trait_values=gq_trait_values,
                    )

    summary_rows = []
    for _, row in df_gq.iterrows():
        geo = row['Geo']
        row_summ = {'geo': geo}
        for t in gq_types:
            row_summ[t] = 0
        for k in ind_codes:
            row_summ['ind_' + k] = 0
        if geo in cbgs:
            cbg_index = cbgs[geo]
            for t_idx, t_code in enumerate(gq_types):
                gq = gqs.get((t_idx + 1, cbg_index))
                if gq is not None:
                    ppl = [gq_people[k] for k in gq.residents]
                    row_summ[t_code] = len(ppl)
                    if t_code == 'ninst1864civ':
                        cats = [p.com_cat if p.com_cat is not None else 0 for p in ppl]
                        for ci, code in enumerate(ind_codes):
                            row_summ['ind_' + code] = sum(1 for c in cats if c == ci + 1)
        summary_rows.append(row_summ)
    gq_summary = pd.DataFrame(summary_rows)

    return cbgs, gqs, gq_people, gq_summary


def _resolve_config(config, data_dir):
    """Use in-memory config when provided; otherwise fall back to data_dir/config.json."""
    if config is not None:
        return config
    return tryJSON(os.path.join(data_dir, 'config.json'))


def _person_columns(p_samps, ind_codes, additional_traits):
    """Precompute the per-person columns needed to build PersonData, as arrays.

    Everything here used to be done row-by-row with ``.iloc``/``.apply(axis=1)``
    inside the person loop, which costs a fresh pandas Series per person. Doing it
    once, column-wise, is several hundred times faster.
    """
    ind_cols = ['ind_' + k for k in ind_codes]
    ind = p_samps[ind_cols].to_numpy(dtype=bool) if ind_cols else np.zeros((len(p_samps), 0), bool)
    has_ind = ind.any(axis=1) if ind.shape[1] else np.zeros(len(p_samps), bool)
    # first_true(row) == argmax for a boolean row, but only where some value is True
    first_ind = ind.argmax(axis=1) if ind.shape[1] else np.zeros(len(p_samps), int)

    commuter = p_samps['commuter'].fillna(False).to_numpy(dtype=bool)
    com_cat = np.where(commuter & has_ind, first_ind + 1, -1)

    income = p_samps[['com_LODES_low', 'com_LODES_high']].to_numpy(dtype=bool)
    has_income = income.any(axis=1)
    com_inc = np.where(has_income, income.argmax(axis=1) + 1, -1)

    sch_grade = p_samps['sch_grade'].to_numpy(dtype=object)
    sch_grade = np.where(pd.isna(sch_grade), None, sch_grade)

    # Traits become a tuple per person, positioned by a schema shared population-wide
    schema = TraitSchema(additional_traits)
    if additional_traits:
        raw = p_samps[list(additional_traits)].to_numpy(dtype=object)
        trait_vals = np.where(pd.isna(raw), None, raw.astype(bool, copy=False)
                              if raw.dtype != object else raw)
        trait_rows = [tuple(None if v is None else bool(v) for v in row) for row in trait_vals]
    else:
        trait_rows = [()] * len(p_samps)

    return dict(
        age=p_samps['AGEP'].to_numpy(dtype=np.int64),
        working=p_samps['has_job'].fillna(False).to_numpy(dtype=bool),
        commuter=commuter,
        com_cat=com_cat,
        com_inc=com_inc,
        sch_grade=sch_grade,
        schema=schema,
        trait_rows=trait_rows,
    )


def generate_people(co_results, data_dir, config=None, random_seed=None):
    rng = np.random.default_rng(random_seed)
    config = _resolve_config(config, data_dir)
    additional_traits = config.get('additional_traits', [])
    wp_codes = tryJSON(os.path.join(data_dir, 'processed', 'codes.json'))
    ind_codes = wp_codes.get('ind_codes', [])

    counties = read_counties(data_dir)
    hh_idx = read_hh_serials(data_dir)

    p_samps = read_psamp_df(data_dir, ind_codes, additional_traits)
    p_idx = people_by_serial(p_samps)
    cols = _person_columns(p_samps, ind_codes, additional_traits)
    age, working, commuter = cols['age'], cols['working'], cols['commuter']
    com_cat, com_inc, sch_grade = cols['com_cat'], cols['com_inc'], cols['sch_grade']
    schema, trait_rows = cols['schema'], cols['trait_rows']

    cbgs = {}
    cbg_indexer = Indexer()
    households = {}
    people = {}

    for c in counties:
        if c not in co_results:
            continue
        cbg_hhs = co_results[c]

        for cbg_code, hh_vec in cbg_hhs.items():
            cbg_i = cbg_indexer(cbgs, cbg_code)
            for hh_i_0, hh_serial in enumerate(hh_vec):
                hh_i = hh_i_0 + 1
                hh_key = (hh_i, cbg_i)
                p_vec = p_idx.get(hh_serial, [])
                for p_i_0, r in enumerate(p_vec):
                    j = r - 1  # p_idx is 1-based for Julia parity
                    grade = sch_grade[j]
                    people[(p_i_0 + 1, hh_i, cbg_i)] = PersonData(
                        hh=hh_key,
                        sample=r,
                        age=int(age[j]),
                        working=bool(working[j]),
                        commuter=bool(commuter[j]),
                        com_cat=int(com_cat[j]) if com_cat[j] > 0 else None,
                        com_inc=int(com_inc[j]) if com_inc[j] > 0 else None,
                        sch_grade=str(grade) if grade is not None else None,
                        schema=schema,
                        trait_values=trait_rows[j],
                    )
                households[hh_key] = Household(
                    sample=hh_idx.get(hh_serial, 0),
                    people=[(i + 1, hh_i, cbg_i) for i in range(len(p_vec))]
                )

    cbgs, gqs, gq_people, gq_summary = generate_group_quarters(
        config, cbgs, cbg_indexer, ind_codes, data_dir, rng, schema=schema)
    people.update(gq_people)
    return cbgs, people, households, gqs, gq_summary
