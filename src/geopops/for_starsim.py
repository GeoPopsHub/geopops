# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Johns Hopkins University
import pandas as pd
import numpy as np
import starsim as ss
from scipy.io import mmread
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Built-in network name -> (matrix file tag, output csv name).
_LAYERS = {
    "homenet": ("hh", "net_h"),
    "schoolnet": ("sch", "net_s"),
    "worknet": ("wp", "net_w"),
    "gqnet": ("gq", "net_g"),
}

# Cached network dataframes for one population at a time, filled in one layer
# at a time as each is requested, with the path key identifying which
# population they came from.
# Asking for a different population reloads. Without that key -- as was the
# case originally -- a second population in the same session silently got
# the first one's edges.
_NETS = {}
_NET_KEY = None
# edge_weight each layer's csv was last saved with, so the csv is rewritten
# only when a call asks for a different weight.
_NET_SAVED = {}


class _SubgroupTracking(ss.Analyzer):
    def __init__(self, subgroup, outcome, name=None, state_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_product = False
        self.subgroup = subgroup
        self.outcome = outcome
        self.state_id = state_id
        self.n_outcome = {}
        if name:
            self.name = name

    def step(self):
        sim = self.sim

        if not self.n_outcome:
            groups = np.unique(sim.people[self.subgroup])
            self.n_outcome = {group: [] for group in groups}

        disease_name = sim.diseases[0].name.lower()
        disease_obj = getattr(sim.people, disease_name, None)

        for group in self.n_outcome.keys():
            if self.state_id is not None:
                count = len(
                    ss.uids(
                        (sim.people[self.subgroup] == group)
                        & (disease_obj[self.outcome] == 1)
                        & (sim.people.state == self.state_id)
                    )
                )
            else:
                count = len(ss.uids((sim.people[self.subgroup] == group) & (disease_obj[self.outcome] == 1)))
            self.n_outcome[group].append(count)

    def get_subgroup_data(self):
        """Return a DataFrame where rows are subgroups and columns are time steps."""
        df = pd.DataFrame.from_dict(self.n_outcome, orient='index')
        df.columns = [f't_{i}' for i in range(len(df.columns))]
        df.index.name = self.subgroup
        df = df.reset_index()
        return df


class _GPNetwork(ss.Network):
    def __init__(self, name, edge_weight=1.0, csv_path=None, network_df=None, p1_col='p1', p2_col='p2', beta_col=None,
                 path=None):
        super().__init__()
        self.name = name
        self.edge_weight = edge_weight
        self.csv_path = csv_path
        self.network_df_input = network_df
        self.p1_col = p1_col
        self.p2_col = p2_col
        self.beta_col = beta_col
        # Which population to read. This is the only value this class ever
        # needed from a config; taking it directly is what lets a caller keep
        # GPNetwork and People in agreement.
        self.path = path

        if self.csv_path is not None and self.network_df_input is not None:
            raise ValueError("Provide only one of csv_path or network_df, not both.")

        if self.network_df_input is not None:
            self.network_df = self._normalize_network_dataframe(self.network_df_input)
        elif self.csv_path is not None:
            self.network_df = self._load_custom_network_csv()
        else:
            if name not in _LAYERS:
                raise ValueError(
                    f"Unknown network name '{name}'. Available built-in names: {list(_LAYERS.keys())}. "
                    "To use a custom CSV network, provide csv_path=..."
                )

            self._ensure_network_created()

            self.network_df = _NETS[name]

        self._populate_edges()

    def _load_custom_network_csv(self):
        """Load and validate a custom edge-list CSV for network creation."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Custom network file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        return self._normalize_network_dataframe(df, source_desc=f"CSV '{self.csv_path}'")

    def _normalize_network_dataframe(self, df, source_desc="provided dataframe"):
        """Validate and normalize custom network data into p1/p2/edge_weight columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"network_df must be a pandas DataFrame, got {type(df)}")

        tmp = df.copy()
        if 'Unnamed: 0' in tmp.columns:
            tmp = tmp.drop(columns=['Unnamed: 0'])

        missing_cols = [c for c in [self.p1_col, self.p2_col] if c not in tmp.columns]
        if missing_cols:
            raise ValueError(
                f"{source_desc} is missing required column(s): {missing_cols}. "
                f"Available columns: {list(tmp.columns)}"
            )

        out = pd.DataFrame({
            'p1': pd.to_numeric(tmp[self.p1_col], errors='coerce'),
            'p2': pd.to_numeric(tmp[self.p2_col], errors='coerce'),
        }).dropna(subset=['p1', 'p2'])

        out['p1'] = out['p1'].astype(np.int64)
        out['p2'] = out['p2'].astype(np.int64)

        if self.beta_col is not None:
            if self.beta_col not in tmp.columns:
                raise ValueError(
                    f"beta_col '{self.beta_col}' not found in {source_desc}. "
                    f"Available columns: {list(tmp.columns)}"
                )
            beta_vals = pd.to_numeric(tmp[self.beta_col], errors='coerce')
            beta_vals = beta_vals.loc[out.index].fillna(float(self.edge_weight))
            out['edge_weight'] = beta_vals.astype(float).values
        else:
            out['edge_weight'] = float(self.edge_weight)

        if out.empty:
            raise ValueError(f"{source_desc} has no valid edges after parsing.")

        return out.reset_index(drop=True)

    def _populate_edges(self):
        """Populate network edges from dataframe."""
        self.edges.p1 = self.network_df['p1'].values
        self.edges.p2 = self.network_df['p2'].values
        # Honor an explicitly provided scalar edge_weight (e.g., 2.0 for homenet)
        # so users can rescale built-in networks without editing CSV/dataframes.
        if float(self.edge_weight) != 1.0:
            self.edges.beta = np.full(len(self.network_df), float(self.edge_weight))
        elif 'edge_weight' in self.network_df.columns:
            self.edges.beta = self.network_df['edge_weight'].values.astype(float)
        else:
            self.edges.beta = np.full(len(self.network_df), self.edge_weight)
        self.validate()

    def _network_config(self):
        """Resolve path -- which population to read.

        An explicitly passed path wins. If not given it falls back to the
        package config, which is what callers that pass nothing have always
        used, so existing calls are unaffected.
        """
        path = self.path
        if path is None:
            cfg_path = os.path.join(BASE_DIR, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"config.json file not found at {cfg_path}")
            with open(cfg_path, "r") as f:
                config = json.load(f)
            path = config.get("path")
        return path

    def _ensure_network_created(self):
        """Load this population's layer for self.name unless it is already cached.

        The layer's csv is saved with self.edge_weight, and rewritten whenever
        a call asks for a different weight than the one last saved.
        """
        global _NETS, _NET_KEY, _NET_SAVED
        key = self._network_config()
        if _NET_KEY != key:
            _NETS, _NET_SAVED, _NET_KEY = {}, {}, key
        if self.name not in _NETS:
            _NETS[self.name] = _create_network(self.name, key)
        weight = float(self.edge_weight)
        if _NET_SAVED.get(self.name) != weight:
            _save_network(self.name, key, _NETS[self.name], weight)
            _NET_SAVED[self.name] = weight

    def step(self):
        self.validate()


def _create_network(name, path):
    """Read one of a population's .mtx layers into a dataframe.

    Edges are kept as stored in the matrix file: p1 is the column index, p2
    the row index.
    """
    print(f"\n*** Running for_starsim.network(): loading {name} matrix ***")
    tag, _ = _LAYERS[name]

    m = mmread(os.path.join(path, "pop_export", f"adj_upper_triang_{tag}.mtx"))
    net = pd.DataFrame({
        "p1": np.asarray(m.col, dtype=np.int64),
        "p2": np.asarray(m.row, dtype=np.int64),
    })
    net["edge_weight"] = np.int64(1)
    return net


def _save_network(name, path, net, edge_weight):
    """Save one layer's edge list to CSV with the requested edge_weight."""
    _, out = _LAYERS[name]
    starsim_dir = os.path.join(path, "pop_export", "starsim")
    os.makedirs(starsim_dir, exist_ok=True)

    # The default weight is written as the integer 1 already in the dataframe,
    # so a default call's csv is unchanged from before edge_weight was saved.
    if edge_weight != 1.0:
        net = net.assign(edge_weight=edge_weight)
    net.to_csv(os.path.join(starsim_dir, f"{out}.csv"), index=False)

    print(f"Network csv file {out}.csv created and saved successfully (edge_weight={edge_weight:g})")


# ---------------------------------------------------------------------------
# Public API. These were methods on a `ForStarsim` class whose __init__ was
# vestigial and whose instance state no method read -- a namespace, not an
# object. As module functions they read the same way as the rest of the
# package: geopops.for_starsim.people(), .network(), .subgroup_tracking().
# ---------------------------------------------------------------------------

def _load_config(config_dict=None, config_path=None, base_dir=None):
    base_dir = base_dir if base_dir is not None else BASE_DIR
    if config_dict is not None:
        return config_dict
    cfg_path = config_path if config_path is not None else os.path.join(base_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"config.json file not found at {cfg_path}. Please create this file with the required configuration."
        )
    with open(cfg_path, "r") as f:
        return json.load(f)


def people(config_dict=None, config_path=None, base_dir=None, path=None):
    """Create and return a Starsim People object directly.

    Args:
        path: population directory (the one holding pop_export/). Overrides
            the config's path. Pass the same value to GPNetwork() so the two
            are guaranteed to read the same population.
    """
    print("\n*** Running for_starsim.people() ***")
    if path is None:
        config = _load_config(config_dict=config_dict, config_path=config_path, base_dir=base_dir)
        path = config.get("path")

    adj_mat_keys = pd.read_csv(f'{path}/pop_export/adj_mat_keys.csv')
    people = pd.read_csv(f'{path}/pop_export/people.csv')
    ppl_df = adj_mat_keys.merge(people, on=['p_id', 'hh_id', 'cbg_id'], how='left')
    schools = pd.read_csv(f'{path}/pop_export/sch_students.csv')
    ppl_df = ppl_df.merge(schools, on=['p_id', 'hh_id', 'cbg_id'], how='left')
    ppl_df.loc[ppl_df['sch_code'].isnull(), 'sch_code'] = 0
    ppl_df.insert(0, 'uid', ppl_df['index_zero'].values)

    cbg_idxs = pd.read_csv(f'{path}/pop_export/cbg_idxs.csv')
    ppl_df = ppl_df.merge(cbg_idxs, on='cbg_id', how='left')
    ppl_df['state'] = ppl_df['cbg_geocode'].astype(str).str[:2].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
    ppl_df['county'] = ppl_df['cbg_geocode'].astype(str).str[:5].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
    ppl_df['tract'] = ppl_df['cbg_geocode'].astype(str).str[:11].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
    ppl_df['cbg_geocode'] = ppl_df['cbg_geocode'].astype(str).str[:12].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
    ppl_df.loc[(ppl_df['age'] >= 0) & (~ppl_df['age'].isnull()), 'agegroup'] = 0.0
    ppl_df.loc[(ppl_df['age'] >= 10) & (~ppl_df['age'].isnull()), 'agegroup'] = 1.0
    ppl_df.loc[(ppl_df['age'] >= 20) & (~ppl_df['age'].isnull()), 'agegroup'] = 2.0
    ppl_df.loc[(ppl_df['age'] >= 30) & (~ppl_df['age'].isnull()), 'agegroup'] = 3.0
    ppl_df.loc[(ppl_df['age'] >= 40) & (~ppl_df['age'].isnull()), 'agegroup'] = 4.0
    ppl_df.loc[(ppl_df['age'] >= 50) & (~ppl_df['age'].isnull()), 'agegroup'] = 5.0
    ppl_df.loc[(ppl_df['age'] >= 60) & (~ppl_df['age'].isnull()), 'agegroup'] = 6.0
    ppl_df.loc[(ppl_df['age'] >= 70) & (~ppl_df['age'].isnull()), 'agegroup'] = 7.0
    ppl_df.loc[(ppl_df['age'] >= 80) & (~ppl_df['age'].isnull()), 'agegroup'] = 8.0
    ppl_df.loc[(ppl_df['age'] >= 90) & (~ppl_df['age'].isnull()), 'agegroup'] = 9.0

    hh = pd.read_csv(f'{path}/pop_export/hh.csv')
    hh['household'] = hh.index + 1
    hh.drop(columns=['sample_index'], inplace=True)
    ppl_df = ppl_df.merge(hh, on=['cbg_id', 'hh_id'], how='left')
    ppl_df.loc[ppl_df['household'].isnull(), 'household'] = 0
    race_trait_cols = [
        'race_white_alone', 'race_black_alone', 'race_amerindian_or_alaskan',
        'race_asian_alone', 'race_pacific_alone', 'race_other_alone',
        'race_two_or_more', 'hispanic',
    ]
    ppl_df = ppl_df[['uid', 'p_id', 'hh_id', 'cbg_id', 'sample_index', 'state', 'county', 'tract', 'cbg_geocode',
                     'household', 'age', 'agegroup', 'female', *race_trait_cols, 'working', 'commuter',
                     'commuter_income_category', 'commuter_workplace_category', 'sch_grade', 'sch_code']]
    ppl_df.to_csv(f'{path}/pop_export/people_all.csv', index=False)

    age = ss.FloatArr('age', default=ss.BaseArr(ppl_df['age'].values))
    agegroup = ss.FloatArr('agegroup', default=ss.BaseArr(ppl_df['agegroup'].values))
    female = ss.FloatArr('female', default=ss.BaseArr(ppl_df['female'].values))
    race_trait_states = [
        ss.FloatArr(col, default=ss.BaseArr(ppl_df[col].values)) for col in race_trait_cols
    ]
    state = ss.IntArr('state', default=ss.BaseArr(ppl_df['state'].values))
    county = ss.IntArr('county', default=ss.BaseArr(ppl_df['county'].values))
    tract = ss.IntArr('tract', default=ss.BaseArr(ppl_df['tract'].values))
    cbg_geocode = ss.IntArr('cbg_geocode', default=ss.BaseArr(ppl_df['cbg_geocode'].values))
    household = ss.IntArr('household', default=ss.BaseArr(ppl_df['household'].values))
    commuter = ss.FloatArr('commuter', default=ss.BaseArr(ppl_df['commuter'].values))
    commuter_income_category = ss.FloatArr(
        'commuter_income_category', default=ss.BaseArr(ppl_df['commuter_income_category'].values)
    )
    commuter_workplace_category = ss.FloatArr(
        'commuter_workplace_category', default=ss.BaseArr(ppl_df['commuter_workplace_category'].values)
    )
    sch_code = ss.IntArr('sch_code', default=ss.BaseArr(ppl_df['sch_code'].values))

    ppl = ss.People(n_agents=len(ppl_df), extra_states=[
        agegroup, *race_trait_states, state, county, tract, cbg_geocode,
        household, commuter, commuter_income_category, commuter_workplace_category, sch_code
    ])

    ppl.states.append(age, overwrite=True)
    setattr(ppl, age.name, age)
    age.link_people(ppl)

    ppl.states.append(female, overwrite=True)
    setattr(ppl, female.name, female)
    female.link_people(ppl)

    sim = ss.Sim(people=ppl).init()
    _ = sim  # keep side-effect parity with previous implementation
    os.makedirs(f'{path}/pop_export/starsim', exist_ok=True)
    ss.save(f'{path}/pop_export/starsim/ppl.pkl', ppl)
    print("Starsim People object created and saved successfully")
    return ppl


def network(name, edge_weight=1.0, csv_path=None, network_df=None, p1_col='p1', p2_col='p2', beta_col=None,
              path=None):
    """Build one of the built-in networks for a population.

    Args:
        path: population directory (the one holding pop_export/). Defaults
            to the package config's path. Pass the same value used for
            people() to guarantee the two read the same population.
    """
    return _GPNetwork(
        name=name,
        edge_weight=edge_weight,
        csv_path=csv_path,
        network_df=network_df,
        p1_col=p1_col,
        p2_col=p2_col,
        beta_col=beta_col,
        path=path,
    )


def subgroup_tracking(subgroup, outcome, name=None, state_id=None, *args, **kwargs):
    return _SubgroupTracking(subgroup, outcome, name=name, state_id=state_id, *args, **kwargs)
