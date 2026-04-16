"""
Minimal tests of running GeoPops with Julia

Prerequisites:
- Julia installed
- .env file with CENSUS_API_KEY and JULIA_ENV_PATH (see .env.example)
"""

import sciris as sc
import geopops

pars_geopops = {'path': "data", # Set a folder where you want output files to be stored
                'main_year': 2019, # Year of data
                'geos': ["45083"], # State or county fips of your geographical location of interest. Example of Spartanburg SC
                'commute_states': ["45","37"], # State fips of commute data to download. Example of SC, NC
                'use_pums': ["45","37"], # State fips of PUMS data to download. Example of SC, NC
                } 

c = geopops.WriteConfig(**pars_geopops) # Define parameters for pop generation in config.json
# c.get_pars() # View all parameters from config.json


@sc.timer()
def test_julia_CO():
    j = geopops.RunJulia()
    j.CO()
    return


@sc.timer()
def test_julia_synthpop():
    j = geopops.RunJulia()
    j.SynthPop()
    return


@sc.timer()
def test_export():
    j = geopops.RunJulia()
    j.Export()
    ppl = geopops.ForStarsim.People()
    h = geopops.ForStarsim.GPNetwork(name='homenet', beta_value=1.0)
    s = geopops.ForStarsim.GPNetwork(name='schoolnet', beta_value=1.0)
    w = geopops.ForStarsim.GPNetwork(name='worknet', beta_value=1.0)
    g = geopops.ForStarsim.GPNetwork(name='gqnet', beta_value=1.0)
    return ppl, h, s, w, g


if __name__ == "__main__":
    T = sc.timer()
    test_julia_CO()
    test_julia_synthpop()
    test_export()
    T.toc()