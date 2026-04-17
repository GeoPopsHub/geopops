import pandas as pd
import numpy as np
import starsim as ss
from scipy.io import mmread
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ForStarsim:
    """Creates and initializes Starsim People objects from GeoPops data.
    
    This class orchestrates the creation of Starsim People objects using processed
    GeoPops data. It follows the same pattern as other GeoPops classes.
    """
    
    # Class-level variables to store network dataframes
    _net_h = None
    _net_s = None
    _net_w = None
    _net_g = None
    
    def __init__(self, config_dict=None, config_path=None, base_dir=None):
        """Create a Starsim runner.
        
        Args:
            config_dict: Optional dict with configuration. If provided, takes precedence over config_path.
            config_path: Optional path to a JSON config file. Defaults to config.json in package directory.
            base_dir: Optional base dir to use for relative paths. Defaults to package directory.
        """
        # Use package directory as base, same as process_data.py
        self.base_dir = base_dir if base_dir is not None else BASE_DIR
        
        if config_dict is not None:
            self.config = config_dict
        else:
            cfg_path = config_path if config_path is not None else os.path.join(self.base_dir, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"config.json file not found at {cfg_path}. Please create this file with the required configuration.")
            with open(cfg_path, "r") as f:
                self.config = json.load(f)
        
        # Set path from config (no hardcoded default)
        self.path = self.config.get("path")
            
    class People:
        def __new__(cls, config_dict=None, config_path=None, base_dir=None):
            """Create and return a Starsim People object directly."""
            instance = super().__new__(cls)
            instance.__init__(config_dict, config_path, base_dir)
            return instance.ppl  # Return the actual Starsim People object directly
            
        def __init__(self, config_dict=None, config_path=None, base_dir=None):
            """Create a Starsim People object from GeoPops data.
            
            Args:
                config_dict: Optional dict with configuration. If provided, takes precedence over config_path.
                config_path: Optional path to a JSON config file. Defaults to config.json in package directory.
                base_dir: Optional base dir to use for relative paths. Defaults to package directory.
            """
            # Use package directory as base, same as process_data.py
            self.base_dir = base_dir if base_dir is not None else BASE_DIR
            
            if config_dict is not None:
                self.config = config_dict
            else:
                cfg_path = config_path if config_path is not None else os.path.join(self.base_dir, "config.json")
                if not os.path.exists(cfg_path):
                    raise FileNotFoundError(f"config.json file not found at {cfg_path}. Please create this file with the required configuration.")
                with open(cfg_path, "r") as f:
                    self.config = json.load(f)
            
            # Set path from config
            self.path = self.config.get("path")
            
            # Create the people object
            self._create_people()
            
        def _create_people(self):
            """Create the Starsim People object from GeoPops data."""
            print("Creating Starsim People object from GeoPops data")
            
            # Make people dataframe
            # Read in all agents, including dummy agents, where index corresponds to agent ID in network adjacency matrices
            adj_mat_keys = pd.read_csv(f'{self.path}/pop_export/adj_mat_keys.csv')
            # Read in agents and attributes for people living in geo area
            people = pd.read_csv(f'{self.path}/pop_export/people.csv')
            # Merge adj_mat_keys to people and attributes
            ppl_df = adj_mat_keys.merge(people, on=['p_id','hh_id','cbg_id'], how='left')
            # Merge school ids into people dataframe
            schools = pd.read_csv(f'{self.path}/pop_export/sch_students.csv')
            ppl_df = ppl_df.merge(schools, on=['p_id','hh_id','cbg_id'], how='left')
            ppl_df.loc[ppl_df['sch_code'].isnull(), 'sch_code'] = 0
            # Insert unique ID for each agent
            ppl_df.insert(0, 'uid', ppl_df['index_zero'].values)
            # Map cbg_id to 15-digit FIPS code
            cbg_idxs = pd.read_csv(f'{self.path}/pop_export/cbg_idxs.csv')
            ppl_df = ppl_df.merge(cbg_idxs, on='cbg_id', how='left')
            # Create columns for state, county, tract, and cbg
            ppl_df['state'] = ppl_df['cbg_geocode'].astype(str).str[:2].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
            ppl_df['county'] = ppl_df['cbg_geocode'].astype(str).str[:5].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
            ppl_df['tract'] = ppl_df['cbg_geocode'].astype(str).str[:11].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
            ppl_df['cbg_geocode'] = ppl_df['cbg_geocode'].astype(str).str[:12].replace({'na': '0.0', 'nan': '0.0'}).astype(float)
            # Create 10-year age groups (dummy agents do not have age data)
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
            # Add household ID
            # Add household ID as a state
            hh = pd.read_csv(f'{self.path}/pop_export/hh.csv')
            hh['household'] = hh.index + 1 # Turn the idex into a unique household id
            hh.drop(columns=['sample_index'], inplace=True)
            ppl_df = ppl_df.merge(hh, on=['cbg_id','hh_id'], how='left') 
            ppl_df.loc[ppl_df['household'].isnull(), 'household'] = 0 # replace NaN with 0
            # Create race/ethnicity categories in single variable
            ppl_df.loc[ppl_df['race_black_alone'] == 0, 'race_ethnicity'] = 0.0 # Black alone
            ppl_df.loc[ppl_df['hispanic'] == 1, 'race_ethnicity'] = 1.0 # Hispanic
            ppl_df.loc[ppl_df['white_non_hispanic'] == 1, 'race_ethnicity'] = 2.0 # White non-Hispanic
            # print(ppl_df.columns)
            # Reorder columns
            ppl_df = ppl_df[['uid','p_id','hh_id','cbg_id','sample_index','state','county','tract','cbg_geocode','household',
                            'age','agegroup','female', 'race_black_alone','white_non_hispanic','hispanic','race_ethnicity',
                            'working','commuter','commuter_income_category','commuter_workplace_category',
                            'sch_grade','sch_code']]
            # Export to csv
            ppl_df.to_csv(f'{self.path}/pop_export/people_all.csv', index=False)
            
            # Create Starsim states
            age = ss.FloatArr('age', default=ss.BaseArr(ppl_df['age'].values))
            agegroup = ss.FloatArr('agegroup', default=ss.BaseArr(ppl_df['agegroup'].values))
            female = ss.FloatArr('female', default=ss.BaseArr(ppl_df['female'].values))
            race_ethnicity = ss.FloatArr('race_ethnicity', default=ss.BaseArr(ppl_df['race_ethnicity'].values))
            state = ss.IntArr('state', default=ss.BaseArr(ppl_df['state'].values))
            county = ss.IntArr('county', default=ss.BaseArr(ppl_df['county'].values))
            tract = ss.IntArr('tract', default=ss.BaseArr(ppl_df['tract'].values))
            cbg_geocode = ss.IntArr('cbg_geocode', default=ss.BaseArr(ppl_df['cbg_geocode'].values))
            household = ss.IntArr('household', default=ss.BaseArr(ppl_df['household'].values)) # Turn this array into a Starsim IntArr
            commuter = ss.FloatArr('commuter', default=ss.BaseArr(ppl_df['commuter'].values))
            commuter_income_category = ss.FloatArr('commuter_income_category', default=ss.BaseArr(ppl_df['commuter_income_category'].values))
            commuter_workplace_category = ss.FloatArr('commuter_workplace_category', default=ss.BaseArr(ppl_df['commuter_workplace_category'].values))
            # sch_grade = ss.StringArr('sch_grade', default=ss.BaseArr(ppl_df['sch_grade'].values))
            sch_code = ss.IntArr('sch_code', default=ss.BaseArr(ppl_df['sch_code'].values))
            
            # Create the people object
            self.ppl = ss.People(n_agents=len(ppl_df), extra_states=[agegroup, 
                                                                     race_ethnicity, 
                                                                     state,
                                                                     county,
                                                                     tract,
                                                                     cbg_geocode,
                                                                     household,
                                                                     commuter,
                                                                     commuter_income_category,
                                                                     commuter_workplace_category,
                                                                     sch_code])
            
            # Add the age state to the existing people object 
            self.ppl.states.append(age, overwrite=True)
            setattr(self.ppl, age.name, age)
            age.link_people(self.ppl)
            
            # Add the female state to the existing people object 
            self.ppl.states.append(female, overwrite=True)
            setattr(self.ppl, female.name, female)
            female.link_people(self.ppl)
            
            # Initialize sim and save ppl object 
            self.sim = ss.Sim(people=self.ppl).init()
            
            # Create output directory if it doesn't exist
            os.makedirs(f'{self.path}/pop_export/starsim', exist_ok=True)
            ss.save(f'{self.path}/pop_export/starsim/ppl.pkl', self.ppl)
            print("Starsim People object created and saved successfully")
    
    class SubgroupTracking(ss.Analyzer):
        def __init__(self, subgroup, outcome, name=None, state_id=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.has_product = False 
            self.subgroup = subgroup  # Store the subgroup parameter
            self.outcome = outcome  # Store the outcome parameter
            self.state_id = state_id
            self.n_outcome = {}  # Initialize empty dict, will be populated in step()
            if name:
                self.name = name  # Set the name for ndict key generation
            return
        
        def step(self):
            sim = self.sim
            
            # Initialize groups on first step if not already done
            if not self.n_outcome:
                groups = np.unique(sim.people[self.subgroup])
                self.n_outcome = {group: [] for group in groups}
                
            # Automatically get disease name from the first disease in sim
            # In starsim, diseases are accessible via their name attribute
            disease_name = sim.diseases[0].name.lower()  # Get name and convert to lowercase
            # Get the disease object dynamically using getattr
            disease_obj = getattr(sim.people, disease_name, None)
                
            # Count n_outcome by subgroup each time step using loop
            for group in self.n_outcome.keys():
                if self.state_id is not None:
                    count = len(ss.uids((sim.people[self.subgroup] == group) & (disease_obj[self.outcome] == 1) & (sim.people.state == self.state_id)))
                else:
                    count = len(ss.uids((sim.people[self.subgroup] == group) & (disease_obj[self.outcome] == 1)))
                self.n_outcome[group].append(count)
            return
            
        def get_subgroup_data(self):  # Renamed from get_cbg_data for clarity
            """Return a DataFrame where rows are subgroups and columns are time steps"""
            df = pd.DataFrame.from_dict(self.n_outcome, orient='index')
            df.columns = [f't_{i}' for i in range(len(df.columns))]
            df.index.name = self.subgroup  # Use self.subgroup
            df = df.reset_index()
            return df
    
    class GPNetwork(ss.Network):
        def __init__(self, name, edge_weight=1.0, csv_path=None, network_df=None, p1_col='p1', p2_col='p2', beta_col=None):
            super().__init__()
            self.name = name
            self.edge_weight = edge_weight
            self.csv_path = csv_path
            self.network_df_input = network_df
            self.p1_col = p1_col
            self.p2_col = p2_col
            self.beta_col = beta_col
            
            if self.csv_path is not None and self.network_df_input is not None:
                raise ValueError("Provide only one of csv_path or network_df, not both.")

            # If a custom dataframe is provided, use it directly.
            if self.network_df_input is not None:
                self.network_df = self._normalize_network_dataframe(self.network_df_input)
            # If a custom CSV is provided, load and normalize it.
            elif self.csv_path is not None:
                self.network_df = self._load_custom_network_csv()
            else:
                # Load config and create networks if not already created
                self._ensure_networks_created()
                
                # Access class-level dataframes
                self.network_map = {
                    'homenet': ForStarsim._net_h,
                    'schoolnet': ForStarsim._net_s, 
                    'worknet': ForStarsim._net_w,
                    'gqnet': ForStarsim._net_g
                }
                
                if name not in self.network_map:
                    raise ValueError(
                        f"Unknown network name '{name}'. Available built-in names: {list(self.network_map.keys())}. "
                        "To use a custom CSV network, provide csv_path=..."
                    )
                
                self.network_df = self.network_map[name]
            
            # Populate edges immediately
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

            # Keep an edge_weight column for consistency with built-in network dataframes.
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
            if 'edge_weight' in self.network_df.columns:
                self.edges.beta = self.network_df['edge_weight'].values.astype(float)
            else:
                self.edges.beta = np.full(len(self.network_df), self.edge_weight)
            self.validate()
            
        def _ensure_networks_created(self):
            """Create networks if they haven't been created yet."""
            if ForStarsim._net_h is None:
                self._create_networks()
                
        def _create_networks(self):
            """Create network dataframes from matrix files."""
            # Load config to get path
            base_dir = BASE_DIR
            cfg_path = os.path.join(base_dir, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"config.json file not found at {cfg_path}")
            with open(cfg_path, "r") as f:
                config = json.load(f)
            path = config.get("path")
            
            def newLayer(file):
                m = mmread(file)
                mat_data = {'p1': m.col, 'p2': m.row}
                mat = pd.DataFrame(data=mat_data)
                mat['edge_weight'] = 1
                return mat
            
            # read in matrix files
            ForStarsim._net_h = newLayer(f'{path}/pop_export/adj_upper_triang_hh.mtx')
            ForStarsim._net_s = newLayer(f'{path}/pop_export/adj_upper_triang_sch.mtx')
            ForStarsim._net_w = newLayer(f'{path}/pop_export/adj_upper_triang_wp.mtx')
            ForStarsim._net_g = newLayer(f'{path}/pop_export/adj_upper_triang_gq.mtx')

            # export csv files
            ForStarsim._net_h.to_csv(f'{path}/pop_export/starsim/net_h.csv') 
            ForStarsim._net_s.to_csv(f'{path}/pop_export/starsim/net_s.csv')
            ForStarsim._net_w.to_csv(f'{path}/pop_export/starsim/net_w.csv')
            ForStarsim._net_g.to_csv(f'{path}/pop_export/starsim/net_g.csv')
            
            print("Network csv files created and saved successfully")
            
        def step(self):
            # Edges are already populated, just validate
            self.validate()
    
def main():
    """Main function for command-line usage."""
    runner = ForStarsim()
    return runner 