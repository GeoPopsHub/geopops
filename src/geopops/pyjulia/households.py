"""
Household, person, and group quarters generation from PUMS samples.
Translated from julia/households.jl.
"""
import numpy as np
import pandas as pd
import os
from .utils import (PersonData, Household, GQres, Indexer, tryJSON,
                    thresh, ranges, first_true, lrRound)


def read_counties(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'cbg_geo.csv'),
                     usecols=['county'], dtype={'county': str})
    return sorted(df['county'].unique().tolist())


def read_hh_serials(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'hh_samples.csv'),
                     usecols=['SERIALNO'], dtype={'SERIALNO': str})
    return dict(zip(df['SERIALNO'], range(len(df))))


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
    """Group person sample row indices by household serial number."""
    result = {}
    for idx, serial in enumerate(p_samps['SERIALNO']):
        result.setdefault(serial, []).append(idx)
    return result


def _row_gq_employment(n, jobtype, ind_codes, row):
    """Number employed by industry for a GQ row."""
    if jobtype == 'none' or n < 1:
        return [0] * len(ind_codes)
    prefix = 'civ_ind_' if jobtype == 'civ' else 'mil_ind_'
    return [int(round(row.get(prefix + k, 0))) for k in ind_codes]


def generate_group_quarters(config, cbgs, cbg_indexer, ind_codes, data_dir, rng):
    """Generate group quarters residents.
    Returns (cbgs, gqs, gq_people, gq_summary_df).
    """
    min_gq_residents = config.get('min_gq_residents', 20)
    add_trait_cols = config.get('additional_traits', [])

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

    # Compute populations for each GQ type
    df_gq['pop_instu18'] = (df_gq['group quarters:under 18'] * df_gq['p_u18_inst']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_inst1864'] = (df_gq['group quarters:18 to 64'] * df_gq['p_18_64_inst']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_ninst1864civ'] = (df_gq['group quarters:18 to 64'] * df_gq['p_18_64_noninst_civil']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_milGQ'] = (df_gq['group quarters:18 to 64'] * df_gq['p_18_64_noninst_mil']).apply(lambda x: thresh(int(round(x)), min_gq_residents))
    df_gq['pop_inst65o'] = (df_gq['group quarters:65 and over'] * df_gq['p_65o_inst']).apply(lambda x: thresh(int(round(x)), min_gq_residents))

    # Read GQ worker counts by industry
    df_civil = pd.read_csv(os.path.join(data_dir, 'processed', 'gq_civilian_workers.csv'), dtype={'Geo': str})
    df_civil.columns = [c.replace('C24030:', 'civ_ind_').replace('C24010:', 'civ_occ_') for c in df_civil.columns]
    df_mil = pd.read_csv(os.path.join(data_dir, 'processed', 'gq_military_workers.csv'), dtype={'Geo': str})
    df_mil.columns = [c.replace('C24030:', 'mil_ind_').replace('C24010:', 'mil_occ_') for c in df_mil.columns]

    # Commuter and income probabilities
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
                        **{trait: None for trait in add_trait_cols}
                    )

    # Build GQ summary dataframe for workplaces module
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


def generate_people(co_results, data_dir, seed=None):
    """Generate people, households, and group quarters from CO results.
    co_results: dict[county -> dict[cbg_code -> list[serial_numbers]]]
    Returns (cbgs, people, households, gqs, gq_summary).
    """
    rng = np.random.default_rng(seed)
    config = tryJSON(os.path.join(data_dir, 'config.json'))
    additional_traits = config.get('additional_traits', [])
    wp_codes = tryJSON(os.path.join(data_dir, 'processed', 'codes.json'))
    ind_codes = wp_codes.get('ind_codes', [])

    counties = read_counties(data_dir)
    hh_idx = read_hh_serials(data_dir)

    print("reading person samples")
    p_samps = read_psamp_df(data_dir, ind_codes, additional_traits)
    p_idx = people_by_serial(p_samps)

    ind_colnames = ['ind_' + k for k in ind_codes]
    ind_col_idxs = {name: i for i, name in enumerate(ind_colnames)}

    # Pre-compute industry category for each person sample
    p_samps['ind_code'] = p_samps[ind_colnames].apply(
        lambda row: first_true(row.values), axis=1)
    p_samps['com_cat'] = p_samps.apply(
        lambda row: (row['ind_code'] + 1) if (row['commuter'] and row['ind_code'] is not None) else None, axis=1)

    # Income categories
    income_cols = ['com_LODES_low', 'com_LODES_high']
    p_samps['income_code'] = p_samps[income_cols].apply(
        lambda row: first_true(row.values), axis=1)
    p_samps['com_inc'] = p_samps['income_code'].apply(lambda x: (x + 1) if x is not None else None)

    cbgs = {}
    cbg_indexer = Indexer()
    households = {}
    people = {}

    print("generating people")
    for c in counties:
        if c not in co_results:
            continue
        print(f"  county {c}")
        cbg_hhs = co_results[c]

        for cbg_code, hh_vec in cbg_hhs.items():
            cbg_i = cbg_indexer(cbgs, cbg_code)
            for hh_i_0, hh_serial in enumerate(hh_vec):
                hh_i = hh_i_0 + 1  # 1-based
                hh_key = (hh_i, cbg_i)
                p_vec = p_idx.get(hh_serial, [])
                for p_i_0, r in enumerate(p_vec):
                    p_i = p_i_0 + 1  # 1-based
                    row = p_samps.iloc[r]
                    trait_kwargs = {}
                    for trait in additional_traits:
                        val = row.get(trait)
                        trait_kwargs[trait] = bool(val) if pd.notna(val) else None
                    sch_grade = row['sch_grade'] if pd.notna(row['sch_grade']) else None
                    people[(p_i, hh_i, cbg_i)] = PersonData(
                        hh=hh_key,
                        sample=r,
                        age=int(row['AGEP']),
                        working=bool(row['has_job']),
                        commuter=bool(row['commuter']),
                        com_cat=int(row['com_cat']) if row['com_cat'] is not None and pd.notna(row['com_cat']) else None,
                        com_inc=int(row['com_inc']) if row['com_inc'] is not None and pd.notna(row['com_inc']) else None,
                        sch_grade=str(sch_grade) if sch_grade is not None else None,
                        **trait_kwargs,
                    )
                households[hh_key] = Household(
                    sample=hh_idx.get(hh_serial, 0),
                    people=[(i + 1, hh_i, cbg_i) for i in range(len(p_vec))]
                )

    print("generating group quarters")
    cbgs, gqs, gq_people, gq_summary = generate_group_quarters(
        config, cbgs, cbg_indexer, ind_codes, data_dir, rng)
    people.update(gq_people)

    print(f"  {len(people)} people, {len(households)} households, {len(gqs)} group quarters")
    return cbgs, people, households, gqs, gq_summary
