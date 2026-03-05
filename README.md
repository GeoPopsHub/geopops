# GeoPops
**Full documentation and tutorials coming soon!**
GeoPops is in development, and we welcome feedback. Please log any issues.

**GeoPops** is a package for generating geographically and demographically realistic synthetic populations for any US Census location using publically available data. Population generation includes three steps:
1. Generate individuals and households using combinatorial optimization (CO)
2. Assign individuals to school and workplace locations using enrollment data and commute flows
3. Connect individuals within locations using graph algorithms

Resulting files include a list of agents with attributes (e.g., age, gender, income) and networks detailing their connections within home, school, workplace, and group quarters (e.g., correctional facilities, nursing homes) locations. GeoPops is meant to produce reasonable approximations of state and county population characteristics with granularity down to the Census Block Group (CBG).   GeoPops builds on a previous package, [GREASYPOP-CO](https://github.com/CDDEP-DC/GREASYPOP-CO/tree/main) (One Health Trust), and incorporates the following changes:
- All code wrapped in convenient Python package that can be pip installed
- Compatibility with Census data beyond 2019 (still developing)
- Automated data downloading
- Users can adjust all config parameters from the front-end
- Class for exporting files compatible with the agent-based modeling software [Starsim](https://starsim.org/) (Institute for Disease Modeling)

## How to use

See [tutorials/MIDAS](https://github.com/ACCIDDA/GeoPops/tree/main/tutorials/MIDAS) for more detailed usage in a Notebook tutorial. Set up and basic usage below.


First, create a **Julia environment** with the dependencies listed below. It may be easiest to store the environment in the same folder you will use for output files. While called with Python commands, combinatorial optimization, school and workplace assignment, and network generation steps occur in Julia to decrease run time. The following terminal commands should work with MacOS. Don't copy the comments; these are just for reference. The Julia website also has download instructions [here](https://julialang.org/install/).
```
cd "YOUR_PATH"
curl -fsSL https://install.julialang.org | sh
juliaup add 1.9.0        # Install Julia 1.9.0
juliaup default 1.9.0    # Make 1.9.0 the default (optional)
julia +1.9.0 --version   # Run that version once
juliaup update           # Update installed versions
julia                    # Launch Julia and see version
Base.active_project()    # Get path where environment is located. Copy this - will need later
]                        # Enter package mode. prompt changes to "(@v1.9) pkg>"
add CSV@0.10.10          # Add required package versions
add DataFrames@1.5.0
add Graphs@1.8.0 
add InlineStrings@1.4.0 
add JSON@0.21.4
add StatsBase@0.33.21
add Distributions@0.25.112
add MatrixMarket@0.4.0
add ProportionalFitting@0.3.0
status                   # View list of packages
```
If you are using Windows, try the following. The Julia website also has download instructions [here](https://julialang.org/install/). Look for "For Windows instructions, click here".
```
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```

You'll also need a **Python environment** with the dependencies listed in the GeoPops `pyproject.toml`. Install GeoPops from [PyPI](https://pypi.org/project/geopops/).
```
pip install geopops
```

Next, obtain a **Census API key** [here](https://api.census.gov/data/key_signup.html), which will be used for pulling Census data. 

Now in a Python or Notebook script, create a dictionary of parameters. Default parameters are stored in a package file called `config.json`. Pass your dictionary into `WriteConfig()` to overwrite config.json with the parameters for your population of interest. Here's an example for Howard County, MD.
```
pars_geopops = {'path': 'YOUR_OUTPUT_DIR', # designate folder for output files
                'census_api_key': "YOUR_CENSUS_API_KEY", 
                'julia_env_path': "YOUR_JULIA_ENV_PATH",
                'main_year': 2019, # year of data
                'geos': ["24027"], # state or county fips code of main geographical area
                'commute_states': ["24"], # fips of commute states to use
                'use_pums': ["24"]} # PUMS states to use

c = geopops.WriteConfig(**pars_geopops) # Overwrite config.json with your parameters
c.get_pars() # View config.json as dictionary
```
The commands below will create your popoulation and store files in the output directory defined above. Downloaded raw data files are stored in the subfolders census, geo, pums, school, and work. Files created in the preprocessing step are stored in the subfolder called processed. The population in jlse format is stored in the subfolder jlse. `Export()` outputs csv versions into the subfolder pop_export. 
```
geopops.DownloadData(auto_run=True)          # Download all Census and other data sources
geopops.ProcessData()                         # Preprocessing for next steps
j = geopops.RunJulia()
j.run_all()                     # Run Julia scripts (much faster than Python). Can also run separately
# j.CO()                        # Combinatorial optimization. Output in jlse folder                    
# j.SynthPop()                  # School/workplace assignment and network generation
# j.Export()                    # Export to csv format
```
The `ForStarsim()` classes has nested classes which can be passed into a Starsim simulation to run a model on your GeoPops popopulation.
```
geopops.ForStarsim.People()             # Returns a Starsim People object
geopops.ForStarsim.GPNetwork()          # Returns a Starsim Network object
geopops.ForStarsim.SubgroupTracking()   # Returns a Starsim Analyzer object for demographic or geographic subgroup tracking
```
## Tutorials
See [tutorials/MIDAS](https://github.com/ACCIDDA/GeoPops/tree/main/tutorials/MIDAS) for more detailed usage in a Notebook tutorial. GeoPops is in development, and we welcome feedback! Please log any issues as you try it out.

## Support
GeoPops development is a collaboration between the following institutions:
* [ACCIDDA](https://accidda.org/)
* [Insight Net](https://insightnet.us/)
* [Johns Hopkins University Center for Systems Science and Engineering](https://systems.jhu.edu/)
* [One Health Trust](https://onehealthtrust.org/)
* [Institute for Disease Modeling](https://www.idmod.org/)
* [Johns Hopkins University Applied Physics laboratory](https://www.jhuapl.edu/)
* [University of Virginia](https://www.virginia.edu/)

## Conferences

### Insight Net Annual Meeting 2026
**Geopops workshop: An open-source, adaptable framework for agent-based modeling on synthetic populations**
In this hands-on workshop, participants use GeoPops to test interventions in response to the mealses outbreak in South Carolina
* Generate a population of Spartenburg county, SC
* Geographically seed initial infections
* Run an SEIR model on the population using Starsim
* Test two quarantine strategies: 1) infected individual only; 2) infected individual and siblings
* Track outcomes by age and Census Block Group

### MIDAS 2025
**GeoPops demonstration: An open-source, adaptable framework for agent-based modeling on synthetic populations**

This [tutorial](https://github.com/ACCIDDA/GeoPops/tree/main/tutorials/MIDAS) accompanies a speed talk and poster at the MIDAS 2025 annual symposium. It demonstrates:
* Making a population of Howard County, MD
* Running a Starsim SIR ABM on the population
* Incorporating endogenous feedback for "school closures"
* Tracking outcomes by demographic and geographic subgroups

For more realistic applications find me at the following conferences:

### Winter Simulation Conference 2025
**GeoPops: An open-source package for generating geographically realistic synthetic populations** (*Presentation*)
* More detailed methodology and usage
* Compare GeoPops to two other open-source synthetic population generators (UrbanPop and Geo-Synthetic-Pop)
  * Make three MD pops (one with each generator)
  * Compare individuals and homes to real Census data
  * Compare network stats of each home, school, and workplace network
  * Run Starsim COVID-19 model on each population
  * Compare simulation results to each other and observed data for first wave of pandemic
  * Discuss usability of each generator

### Epidemics 2025
**Modeling the impact of dynamic decision making on infectious disease outcomes by demographic and geographic subgroups: An open-source agent-based modeling framework** (*[Poster](https://github.com/ACCIDDA/GeoPops/blob/main/tutorials/Epidemics/Hamilton_Epidemics_Poster.pdf)*)
* Run Starsim COVID-19 model with utility framework on GeoPops population of MD
* Individuals weigh health-wealth trade-offs of staying home from work as a function of their income and age
* Test different policy scenarios (e.g., paid sick leave, cash transfer) and compare to observed data from first wave of pandemic
* Demographic and geographic subgroup tracking for disease and economic outcomes

