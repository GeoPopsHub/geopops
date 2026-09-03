"""Bridge from a generated GeoPops population to Starsim objects.

The public surface is a few functions --- :func:`to_starsim_people`,
:func:`starsim_network`, :func:`starsim_networks` --- plus the
:class:`SubgroupTracking` analyzer. Each takes an explicit population directory, so
nothing is cached across calls and generating two populations in one session gives
two independent sets of networks.
"""
import os
import json

import numpy as np
import pandas as pd
import starsim as ss
from scipy.io import mmread

from .exceptions import ConfigError, DataError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Layer name -> matrix-market file written by :func:`geopops.export.export_networks`
NETWORK_FILES = {
    'homenet': 'adj_upper_triang_hh.mtx',
    'schoolnet': 'adj_upper_triang_sch.mtx',
    'worknet': 'adj_upper_triang_wp.mtx',
    'gqnet': 'adj_upper_triang_gq.mtx',
}

#: Columns of people.csv that are core demographics rather than config-driven traits
_CORE_PERSON_COLS = (
    'p_id', 'hh_id', 'cbg_id', 'sample_index', 'age', 'working', 'commuter',
    'commuter_income_category', 'commuter_workplace_category', 'sch_grade',
)


def _resolve_pop_export(pop_export_dir=None, config=None, config_path=None, base_dir=None):
    """Work out which ``pop_export`` directory to read from."""
    if pop_export_dir is not None:
        return pop_export_dir
    if config is None:
        cfg_path = config_path or os.path.join(base_dir or BASE_DIR, "config.json")
        if not os.path.exists(cfg_path):
            raise ConfigError(
                f"No config found at {cfg_path}. Pass pop_export_dir=... or config=... "
                "to say where the generated population lives."
            )
        with open(cfg_path) as f:
            config = json.load(f)
    path = config.get("path")
    if not path:
        raise ConfigError("config has no 'path' entry, so the population directory is unknown.")
    return os.path.join(path, "pop_export")


def _read(pop_export_dir, name, **kwargs):
    path = os.path.join(pop_export_dir, name)
    if not os.path.exists(path):
        raise DataError(f"Expected exported population file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _load_age_by_matrix_index(pop_export_dir):
    """Map matrix row/col index (``index_zero``) to age, same merge as :func:`to_starsim_people`."""
    adj = _read(pop_export_dir, "adj_mat_keys.csv", low_memory=False)
    people = _read(pop_export_dir, "people.csv", low_memory=False)
    merged = adj.merge(people, on=["p_id", "hh_id", "cbg_id"], how="left")
    merged = merged.drop_duplicates(subset=["index_zero"], keep="first")
    return merged.set_index("index_zero")["age"]


def _canonicalize_undirected_edges_df(net_df, age_by_idx):
    """Reorder ``p1``, ``p2`` so ``age(p1) <= age(p2)`` when both ages exist; else smaller index is ``p1``."""
    if net_df.empty:
        return net_df.copy()
    out = net_df.copy()
    p1 = out["p1"].to_numpy(dtype=np.int64, copy=True)
    p2 = out["p2"].to_numpy(dtype=np.int64, copy=True)
    a1 = age_by_idx.reindex(p1).to_numpy()
    a2 = age_by_idx.reindex(p2).to_numpy()
    a1 = np.where(pd.isna(a1), np.nan, np.asarray(a1, dtype=float))
    a2 = np.where(pd.isna(a2), np.nan, np.asarray(a2, dtype=float))
    both = np.isfinite(a1) & np.isfinite(a2)
    swap = np.zeros(len(out), dtype=bool)
    swap[both] = (a1[both] > a2[both]) | ((a1[both] == a2[both]) & (p1[both] > p2[both]))
    swap[~both] = p1[~both] > p2[~both]
    out.loc[swap, ["p1", "p2"]] = np.column_stack([p2[swap], p1[swap]])
    return out


def _random_flip_undirected_edges_df(net_df, rng):
    """Randomly swap ``(p1, p2)`` per edge with probability 0.5.

    Only changes endpoint labelling, so plots that read ``(p1_age, p2_age)`` as an
    ordered pair do not come out looking triangular.
    """
    if net_df.empty:
        return net_df.copy()
    out = net_df.copy()
    p1 = out["p1"].to_numpy(dtype=np.int64, copy=True)
    p2 = out["p2"].to_numpy(dtype=np.int64, copy=True)
    flip = rng.random(len(out)) < 0.5
    out["p1"] = np.where(flip, p2, p1)
    out["p2"] = np.where(flip, p1, p2)
    return out


def load_network_edges(pop_export_dir, names=None, seed=0, save=True, verbose=1):
    """Read exported adjacency matrices into edge-list dataframes.

    Args:
        pop_export_dir: the run's ``pop_export`` directory.
        names: layer names to load; defaults to all of :data:`NETWORK_FILES`.
        seed: seed for the endpoint-order shuffle.
        save: also write ``pop_export/starsim/net_*.csv``.
        verbose: if truthy, log progress.

    Returns:
        dict: layer name -> dataframe with ``p1``, ``p2``, ``edge_weight``.
    """
    names = list(NETWORK_FILES) if names is None else list(names)
    rng = np.random.default_rng(seed)
    short = {'homenet': 'h', 'schoolnet': 's', 'worknet': 'w', 'gqnet': 'g'}

    if save:
        os.makedirs(os.path.join(pop_export_dir, "starsim"), exist_ok=True)

    edges = {}
    for name in names:
        path = os.path.join(pop_export_dir, NETWORK_FILES[name])
        if not os.path.exists(path):
            raise DataError(f"Network file not found: {path}. Has Export() been run?")
        m = mmread(path)
        df = pd.DataFrame({
            "p1": np.asarray(m.col, dtype=np.int64),
            "p2": np.asarray(m.row, dtype=np.int64),
        })
        df["edge_weight"] = np.int64(1)
        # Treat undirected layers as having ~50/50 endpoint ordering.
        df = _random_flip_undirected_edges_df(df, rng)
        edges[name] = df
        if save:
            out = os.path.join(pop_export_dir, "starsim", f"net_{short.get(name, name)}.csv")
            df.to_csv(out, index=False)
            if verbose:
                print(f"-- {out}")
    return edges


def to_starsim_people(pop_export_dir=None, *, config=None, config_path=None,
                      base_dir=None, save=True, verbose=1):
    """Build a Starsim ``People`` object from an exported GeoPops population.

    Args:
        pop_export_dir: the run's ``pop_export`` directory. If omitted, it is
            derived from ``config`` (or a config file).
        save: also write ``people_all.csv`` and ``starsim/ppl.pkl``.
        verbose: if truthy, log progress.

    Returns:
        ss.People: agents carrying age, sex, geography and the run's traits.
    """
    pop_export_dir = _resolve_pop_export(pop_export_dir, config, config_path, base_dir)
    if verbose:
        print("\n*** Building Starsim People from GeoPops population ***")

    people = _read(pop_export_dir, 'people.csv')
    ppl_df = _read(pop_export_dir, 'adj_mat_keys.csv').merge(
        people, on=['p_id', 'hh_id', 'cbg_id'], how='left')
    ppl_df = ppl_df.merge(_read(pop_export_dir, 'sch_students.csv'),
                          on=['p_id', 'hh_id', 'cbg_id'], how='left')
    ppl_df.loc[ppl_df['sch_code'].isnull(), 'sch_code'] = 0
    ppl_df.insert(0, 'uid', ppl_df['index_zero'].values)

    ppl_df = ppl_df.merge(_read(pop_export_dir, 'cbg_idxs.csv'), on='cbg_id', how='left')

    # Geography: split the 12-digit CBG geocode into its nested levels.
    geocode = ppl_df['cbg_geocode'].astype(str)
    for col, width in (('state', 2), ('county', 5), ('tract', 11), ('cbg_geocode', 12)):
        ppl_df[col] = geocode.str[:width].replace({'na': '0.0', 'nan': '0.0'}).astype(float)

    # Decadal age groups, capped at 90+
    ppl_df['agegroup'] = np.clip(ppl_df['age'] // 10, 0, 9)

    hh = _read(pop_export_dir, 'hh.csv')
    hh['household'] = hh.index + 1
    hh = hh.drop(columns=['sample_index'])
    ppl_df = ppl_df.merge(hh, on=['cbg_id', 'hh_id'], how='left')
    ppl_df.loc[ppl_df['household'].isnull(), 'household'] = 0

    # Trait columns are whatever this run's config asked for, so take them from the
    # exported file rather than a hardcoded list.
    trait_cols = [c for c in people.columns if c not in _CORE_PERSON_COLS]

    ppl_df = ppl_df[['uid', 'p_id', 'hh_id', 'cbg_id', 'sample_index', 'state', 'county',
                     'tract', 'cbg_geocode', 'household', 'age', 'agegroup', *trait_cols,
                     'working', 'commuter', 'commuter_income_category',
                     'commuter_workplace_category', 'sch_grade', 'sch_code']]
    if save:
        ppl_df.to_csv(os.path.join(pop_export_dir, 'people_all.csv'), index=False)

    def farr(col):
        return ss.FloatArr(col, default=ss.BaseArr(ppl_df[col].values))

    def iarr(col):
        return ss.IntArr(col, default=ss.BaseArr(ppl_df[col].values))

    age = farr('age')
    # 'female' is conventionally a trait, but Starsim People expects it as a state
    sex_states = [farr('female')] if 'female' in trait_cols else []
    other_traits = [farr(c) for c in trait_cols if c != 'female']

    extra = [farr('agegroup'), *other_traits,
             iarr('state'), iarr('county'), iarr('tract'), iarr('cbg_geocode'),
             iarr('household'), farr('commuter'),
             farr('commuter_income_category'), farr('commuter_workplace_category'),
             iarr('sch_code')]

    ppl = ss.People(n_agents=len(ppl_df), extra_states=extra)

    # age and female are built into ss.People, so overwrite rather than add
    for state in [age, *sex_states]:
        ppl.states.append(state, overwrite=True)
        setattr(ppl, state.name, state)
        state.link_people(ppl)

    # Initializing a Sim wires up the People object (links states, sets uids)
    ss.Sim(people=ppl).init()

    if save:
        starsim_dir = os.path.join(pop_export_dir, 'starsim')
        os.makedirs(starsim_dir, exist_ok=True)
        ss.save(os.path.join(starsim_dir, 'ppl.pkl'), ppl)
    if verbose:
        print(f"Starsim People created: {len(ppl_df)} agents, traits: {trait_cols}")
    return ppl


class GPNetwork(ss.Network):
    """A Starsim network layer backed by a GeoPops contact layer.

    Either name a built-in layer (``homenet``, ``schoolnet``, ``worknet``, ``gqnet``)
    together with the ``pop_export`` directory it lives in, or supply your own edges
    via ``csv_path=`` or ``network_df=``.
    """

    def __init__(self, name, edge_weight=1.0, pop_export_dir=None, csv_path=None,
                 network_df=None, p1_col='p1', p2_col='p2', beta_col=None,
                 config=None, base_dir=None, seed=0, edges=None, save=False,
                 verbose=0):
        super().__init__()
        self.name = name
        self.edge_weight = edge_weight
        self.p1_col = p1_col
        self.p2_col = p2_col
        self.beta_col = beta_col

        if csv_path is not None and network_df is not None:
            raise ValueError("Provide only one of csv_path or network_df, not both.")

        if network_df is not None:
            self.network_df = self._normalize(network_df)
        elif csv_path is not None:
            if not os.path.exists(csv_path):
                raise DataError(f"Custom network file not found: {csv_path}")
            self.network_df = self._normalize(pd.read_csv(csv_path), f"CSV '{csv_path}'")
        else:
            if name not in NETWORK_FILES:
                raise ValueError(
                    f"Unknown network name {name!r}. Built-in names: {list(NETWORK_FILES)}. "
                    "To use a custom network, pass csv_path=... or network_df=..."
                )
            if edges is None:
                pop_export_dir = _resolve_pop_export(pop_export_dir, config, None, base_dir)
                edges = load_network_edges(pop_export_dir, names=[name], seed=seed,
                                           save=save, verbose=verbose)
            self.network_df = edges[name]

        self._populate_edges()

    def _normalize(self, df, source_desc="provided dataframe"):
        """Validate and normalize custom network data into p1/p2/edge_weight columns."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"network_df must be a pandas DataFrame, got {type(df)}")

        tmp = df.drop(columns=['Unnamed: 0'], errors='ignore')
        missing = [c for c in (self.p1_col, self.p2_col) if c not in tmp.columns]
        if missing:
            raise ValueError(
                f"{source_desc} is missing required column(s): {missing}. "
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
                    f"beta_col {self.beta_col!r} not found in {source_desc}. "
                    f"Available columns: {list(tmp.columns)}"
                )
            beta = pd.to_numeric(tmp[self.beta_col], errors='coerce')
            out['edge_weight'] = beta.loc[out.index].fillna(float(self.edge_weight)).astype(float).values
        else:
            out['edge_weight'] = float(self.edge_weight)

        if out.empty:
            raise ValueError(f"{source_desc} has no valid edges after parsing.")
        return out.reset_index(drop=True)

    def _populate_edges(self):
        self.edges.p1 = self.network_df['p1'].values
        self.edges.p2 = self.network_df['p2'].values
        # An explicitly provided scalar edge_weight (e.g. 2.0 for homenet) wins, so
        # users can rescale built-in networks without editing CSVs or dataframes.
        if float(self.edge_weight) != 1.0:
            self.edges.beta = np.full(len(self.network_df), float(self.edge_weight))
        elif 'edge_weight' in self.network_df.columns:
            self.edges.beta = self.network_df['edge_weight'].values.astype(float)
        else:
            self.edges.beta = np.full(len(self.network_df), self.edge_weight)
        self.validate()

    def step(self):
        self.validate()


def starsim_network(name, pop_export_dir=None, edge_weight=1.0, *, save=False,
                    verbose=0, **kwargs):
    """Build one Starsim network layer from a GeoPops population.

    Args:
        name: one of :data:`NETWORK_FILES`, or any name if `csv_path`/`network_df`
            is given.
        pop_export_dir: the run's ``pop_export`` directory.
        edge_weight: scalar transmission weight applied to every edge.
        save: also write ``pop_export/starsim/net_*.csv``.
        verbose: if truthy, log progress.

    Returns:
        GPNetwork: a Starsim network layer.

    To build all four layers at once, and read each matrix file only once, use
    :func:`starsim_networks`.
    """
    return GPNetwork(name, edge_weight=edge_weight, pop_export_dir=pop_export_dir,
                     save=save, verbose=verbose, **kwargs)


def starsim_networks(pop_export_dir=None, names=None, edge_weight=1.0, seed=0,
                     save=True, verbose=1, **kwargs):
    """Build all GeoPops network layers, reading each matrix file exactly once.

    Returns:
        list[GPNetwork]: one layer per name, ready to pass to ``ss.Sim(networks=...)``.
    """
    pop_export_dir = _resolve_pop_export(pop_export_dir, kwargs.pop('config', None),
                                         None, kwargs.pop('base_dir', None))
    names = list(NETWORK_FILES) if names is None else list(names)
    edges = load_network_edges(pop_export_dir, names=names, seed=seed,
                               save=save, verbose=verbose)
    return [GPNetwork(name, edge_weight=edge_weight, edges=edges, **kwargs) for name in names]


class SubgroupTracking(ss.Analyzer):
    """Track counts of a disease outcome over time, split by an agent attribute.

    Args:
        subgroup: name of the ``People`` state to group by (e.g. ``'agegroup'``).
        outcome: name of the disease state to count (e.g. ``'infected'``).
        state_id: optional additional filter on ``people.state``.
    """

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
            self.n_outcome = {group: [] for group in np.unique(sim.people[self.subgroup])}

        disease_name = sim.diseases[0].name.lower()
        disease_obj = getattr(sim.people, disease_name, None)

        for group in self.n_outcome:
            match = (sim.people[self.subgroup] == group) & (disease_obj[self.outcome] == 1)
            if self.state_id is not None:
                match = match & (sim.people.state == self.state_id)
            self.n_outcome[group].append(len(ss.uids(match)))

    def get_subgroup_data(self):
        """Return a DataFrame where rows are subgroups and columns are time steps."""
        df = pd.DataFrame.from_dict(self.n_outcome, orient='index')
        df.columns = [f't_{i}' for i in range(len(df.columns))]
        df.index.name = self.subgroup
        return df.reset_index()
