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

## Get involved
GeoPops is in development and we welcome feedback! Get in touch if you've tried making a population of your own or want to become a member. You can upload your own example as a respository in the hub.

## Tutorial
See the repo [sc_spartanburg_measles](https://github.com/GeoPopsHub/sc_spartanburg_measles) for a detailed example of how to build a population, simulate a disease, test out interventions, and track outcomes by subgroup. [`1_run_geopops.ipynb`](https://github.com/GeoPopsHub/sc_spartanburg_measles/blob/main/1_run_geopops.ipynb) has instructions on how to build a GeoPops population.

## Support
GeoPops is a collaboration between the following institutions:
* [ACCIDDA](https://accidda.org/)
* [Insight Net](https://insightnet.us/)
* [Johns Hopkins University Center for Systems Science and Engineering](https://systems.jhu.edu/)
* [One Health Trust](https://onehealthtrust.org/)
* [Institute for Disease Modeling](https://www.idmod.org/)
* [Johns Hopkins University Applied Physics laboratory](https://www.jhuapl.edu/)
* [University of Virginia](https://www.virginia.edu/)