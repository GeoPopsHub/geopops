"""
Minimal tests of running GeoPops with Julia

Prerequisites:
- Julia installed
- .env file with CENSUS_API_KEY and JULIA_ENV_PATH (see .env.example)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve().parent.parent / "src"))
import geopops

pars_geopops = {'path': "data", # Set a folder where you want output files to be stored
                'main_year': 2019, # Year of data
                'geos': ["45083"], # State or county fips of your geographical location of interest. Example of Spartanburg SC
                'commute_states': ["45","37"], # State fips of commute data to download. Example of SC, NC
                'use_pums': ["45","37"], # State fips of PUMS data to download. Example of SC, NC
                } 

c = geopops.WriteConfig(**pars_geopops) # Define parameters for pop generation in config.json
# c.get_pars() # View all parameters from config.json

def test_download():
    """ Check that download works (~10 min)"""
    d = geopops.DownloadData(auto_run=True)
    return d

def test_processing():
    """ Check that data processing works (~5 min)"""
    p = geopops.ProcessData(auto_run=True) # auto_run=True to run all 
    return p

def test_julia_CO():
    j = geopops.RunJulia()
    j.CO()
    return

def test_julia_synthpop():
    j = geopops.RunJulia()
    j.SynthPop()
    return

def test_export():
    j = geopops.RunJulia()
    j.Export()
    ppl = geopops.ForStarsim.People()
    h = geopops.ForStarsim.GPNetwork(name='homenet', edge_weight=1.0)
    s = geopops.ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
    w = geopops.ForStarsim.GPNetwork(name='worknet', edge_weight=1.0)
    g = geopops.ForStarsim.GPNetwork(name='gqnet', edge_weight=1.0)
    return ppl, h, s, w, g