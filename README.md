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

There are many packages for generating agents and households from Census data, but GeoPops is the only one that is completely open source and generalizable that spatially connects agents in school and workplace networks. In combination with agent-based modeling tools like Starsim, GeoPops can facilitate timely context-specific scenario modeling of respiratory infectious diseases.

## Get involved
GeoPops is in development and we welcome feedback! Get in touch if you've tried making a population of your own or want to become a member. You can upload your own example as a respository in the [GeoPopsHub](https://github.com/GeoPopsHub).

## Installation

```bash
pip install geopops
```

## How to use

```python
import geopops as gp

# 1. Describe the population you want
cfg = gp.make_config(
    path="data",                    # where downloads and results are written
    geos=["45083"],                 # state or county FIPS (Spartanburg County, SC)
    main_year=2019,
    commute_states=["45", "37"],    # states whose commute data to download (SC, NC)
    use_pums=["45", "37"],          # states whose PUMS samples to draw from
    random_seed=42,                 # set this for reproducible runs
)

# 2. Fetch and prepare the input data (slow; both steps cache to `path`)
gp.download_data(cfg)
gp.process_data(cfg)

# 3. Generate the population
pop = gp.generate_pop(cfg, seed=42)

# 4. Hand it to Starsim
ppl = gp.to_starsim_people(pop.pop_export_dir)
networks = gp.starsim_networks(pop.pop_export_dir)
```

Or run the whole thing at once:

```python
pop = gp.run(cfg, seed=42)
```

A `CENSUS_API_KEY` is required for the download step. Put it in a `.env` file (see [`.env.example`](.env.example)) or set it in the environment; it is read automatically and is never written into the package.

Results land in `<path>/pop_export/`: `people.csv` and `hh.csv` describe the agents and their households, and the `adj_upper_triang_*.mtx` files hold the household, school, workplace, and group-quarters contact networks.

[`1_run_geopops.ipynb`](https://github.com/GeoPopsHub/sc_spartanburg_measles/blob/main/1_run_geopops.ipynb) walks through building a population. See [sc_spartanburg_measles](https://github.com/GeoPopsHub/sc_spartanburg_measles) for a full example that builds a population, simulates a disease, tests interventions, and tracks outcomes by subgroup.

## License

GeoPops is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE). It builds on [GREASYPOP-CO](https://github.com/CDDEP-DC/GREASYPOP-CO) (Copyright 2023 Alexander Tulchinsky), which is AGPL-3.0-or-later; see [NOTICE](NOTICE) for full attribution.

## Support
GeoPops is a collaboration between the following institutions:
* [ACCIDDA](https://accidda.org/)
* [Insight Net](https://insightnet.us/)
* [Johns Hopkins University Center for Systems Science and Engineering](https://systems.jhu.edu/)
* [One Health Trust](https://onehealthtrust.org/)
* [Institute for Disease Modeling](https://www.idmod.org/)
* [Johns Hopkins University Applied Physics laboratory](https://www.jhuapl.edu/)
* [University of Virginia](https://www.virginia.edu/)
