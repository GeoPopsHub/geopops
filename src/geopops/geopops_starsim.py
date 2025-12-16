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
        # Use package directory as base, same as census.py
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
            # Use package directory as base, same as census.py
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
            adj_mat_keys = pd.read_csv(f'{self.path}/pop_export/adj_mat_keys.csv')
            people = pd.read_csv(f'{self.path}/pop_export/people.csv')
            ppl_df = adj_mat_keys.merge(people, on=['p_id','hh_id','cbg_id'], how='left')
            
            # Age groups
            ppl_df.loc[ppl_df['age'] >= 0, 'agegroup'] = 0.0
            ppl_df.loc[ppl_df['age'] >= 10, 'agegroup'] = 1.0
            ppl_df.loc[ppl_df['age'] >= 20, 'agegroup'] = 2.0
            ppl_df.loc[ppl_df['age'] >= 30, 'agegroup'] = 3.0
            ppl_df.loc[ppl_df['age'] >= 40, 'agegroup'] = 4.0
            ppl_df.loc[ppl_df['age'] >= 50, 'agegroup'] = 5.0
            ppl_df.loc[ppl_df['age'] >= 60, 'agegroup'] = 6.0
            ppl_df.loc[ppl_df['age'] >= 70, 'agegroup'] = 7.0
            ppl_df.loc[ppl_df['age'] >= 80, 'agegroup'] = 8.0
            ppl_df.loc[ppl_df['age'] >= 90, 'agegroup'] = 9.0
            
            # Student status
            ppl_df.loc[~ppl_df['sch_grade'].isna(), 'student'] = 1.0
            
            # Race/ethnicity
            ppl_df.loc[ppl_df['race_black_alone'] == 0, 'race'] = 0.0
            ppl_df.loc[ppl_df['hispanic'] == 1, 'race'] = 1.0
            ppl_df.loc[ppl_df['white_non_hispanic'] == 1, 'race'] = 2.0
            
            # State/county
            cbg_idxs = pd.read_csv(f'{self.path}/pop_export/cbg_idxs.csv')
            ppl_df = ppl_df.merge(cbg_idxs, on='cbg_id', how='left')
            # print(ppl_df['cbg_geocode'].unique())
            ppl_df['state'] = ppl_df['cbg_geocode'].astype(str).str[:2].replace('na','0').astype(float)
            ppl_df['county'] = ppl_df['cbg_geocode'].astype(str).str[:5].replace('na','0').astype(float)
            # print(ppl_df['cbg_geocode'].dtype)
            # ppl_df['cbg'] = ppl_df['cbg_geocode'].astype(float).astype(str).str[:12].replace('na','0').astype(float)
            # print(len(ppl_df['cbg'].unique()))
            
            # Income category and disease vulnerability
            # Income, 1 is <=40k, 2 is >40k, 0 is null/not commuter
            # ppl_df['ses'] = ppl_df['commuter_income_category'].fillna(0.0)
            ppl_df.loc[ppl_df['commuter'] == 1, 'ses'] = ppl_df['commuter_income_category'].fillna(0.0)
            
            # Disease vulnerability, 1 is <= age 65, 2 is > age 65, 0 is null/no age data
            ppl_df.loc[ppl_df['age'] <= 65, 'vul'] = 1.0
            ppl_df.loc[ppl_df['age'] >  65, 'vul'] = 2.0
            ppl_df['age'] = ppl_df['age'].fillna(0.0)
            
            # Export to csv
            ppl_df.to_csv(f'{self.path}/pop_export/people_all.csv', index=False)
            
            # Create Starsim states
            age = ss.FloatArr('age', default=ss.BaseArr(ppl_df['age'].values))
            female = ss.FloatArr('female', default=ss.BaseArr(ppl_df['female'].values))
            agegroup = ss.FloatArr('agegroup', default=ss.BaseArr(ppl_df['agegroup'].values))
            race = ss.FloatArr('race', default=ss.BaseArr(ppl_df['race'].values))
            cbg_id = ss.FloatArr('cbg_id', default=ss.BaseArr(ppl_df['cbg_id'].values))
            state = ss.FloatArr('state', default=ss.BaseArr(ppl_df['state'].values))
            county = ss.FloatArr('county', default=ss.BaseArr(ppl_df['county'].values))
            wrk = ss.FloatArr('wrk', default=ss.BaseArr(ppl_df['commuter'].values)) # only commuters have income categories
            worker = ss.FloatArr('worker', default=ss.BaseArr(ppl_df['commuter'].values)) # only commuters have income categories
            student = ss.FloatArr('student', default=ss.BaseArr(ppl_df['student'].values))
            ses = ss.FloatArr('ses', default=ss.BaseArr(ppl_df['ses'].values))
            vul = ss.FloatArr('vul', default=ss.BaseArr(ppl_df['vul'].values))
            hsl = ss.FloatArr('hsl', default=np.random.choice([0.0, 1.0,2.0], size=len(ppl_df), p=[1/3, 1/3, 1/3]))
            
            # Create the people object
            self.ppl = ss.People(n_agents=len(ppl_df), extra_states=[agegroup, race, cbg_id, state, county, wrk, worker, ses, vul, hsl])
            
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
        def __init__(self, name, beta_value=1.0):
            super().__init__()
            self.name = name
            self.beta_value = beta_value
            
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
                raise ValueError(f"Unknown network name '{name}'. Available names: {list(self.network_map.keys())}")
            
            self.network_df = self.network_map[name]
            
            # Populate edges immediately
            self._populate_edges()
            
        def _populate_edges(self):
            """Populate network edges from dataframe."""
            self.edges.p1 = self.network_df['p1'].values
            self.edges.p2 = self.network_df['p2'].values
            self.edges.beta = np.full(len(self.network_df['beta']), self.beta_value)
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
                mat['beta'] = 1
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