---
title: 'GeoPops: A Python package for generating geographically-realistic synthetic populations and networks'
tags:
  - Python
  - Synthetic populations
  - Simulation modeling
  - Infectious disease
  - Spatial activity networks
authors:
  - name: Alisa Hamilton
    orcid: 0000-0001-8783-9383
    corresponding: true
    affiliation: "1, 2"
  - name: Cliff C. Kerr
    affiliation: 3
  - name: Alexander Tulchinsky
    orcid: 0000-0002-7323-5041
    affiliation: 4
  - name: Gary Lin
    orcid: 0000-0002-2269-1135
    affiliation: 5
  - name: Eili Klein
    orcid: 0000-0002-1304-5289
    affiliation: "4, 6"
  - name: Lauren Gardner
    orcid: 0000-0003-1083-3850
    affiliation: "1, 2"
affiliations:
  - name: Johns Hopkins University Dept. of Civil and Systems Engineering
    index: 1
  - name: Johns Hopkins University Center for Systems Science and Engineering
    index: 2
  - name: Institute for Disease Modeling, Gates Foundation
    index: 3
  - name: One Health Trust
    index: 4
  - name: Johns Hopkins University Applied Physics Laboratory
    index: 5
  - name: Johns Hopkins University Dept. of Emergency Medicine
    index: 6
date: 24 September 2026
bibliography: paper.bib
header-includes: |
  ```{=latex}
  \usepackage{colortbl}
  % Table helpers, used inside \begingroup ... \endgroup around a table:
  % \GPwidetable    extends the table 4.5cm into the empty left margin
  %                 (JOSS marginpar) and keeps the caption in the text column.
  % \GPstripedtable shades every other body row light gray; striping starts
  %                 after the header/footer definitions (\endlastfoot).
  \newcommand{\GPwidetable}{%
    \setlength{\LTleft}{-4.5cm}\setlength{\LTright}{0pt plus 1fill}%
    \setlength{\linewidth}{\dimexpr\textwidth+4.5cm\relax}%
    \captionsetup{margin={0pt,4.5cm}}}
  \makeatletter
  \newcount\GProw
  \newif\ifGPstripe
  \newcommand{\GPstripedtable}{%
    \global\GPstripefalse
    \GPclearrowcolor\aftergroup\GPclearrowcolor % don't leak shading into other tables
    \let\GProwcr\LT@tabularcr
    \def\LT@tabularcr{\GProwcr\GPafterrow}%
    \let\GPendlastfoot\endlastfoot
    \def\endlastfoot{\GPendlastfoot\noalign{\global\GPstripetrue\global\GProw=0 }}}
  % \GPneedspace{len}: start a new page unless at least len of space is left
  % (same as \Needspace from the needspace package, which JOSS doesn't ship)
  \newcommand{\GPneedspace}[1]{\par\penalty-100\begingroup
    \setlength{\dimen@}{#1}\dimen@ii\pagegoal\advance\dimen@ii-\pagetotal
    \ifdim\dimen@>\dimen@ii\ifdim\dimen@ii>\z@\vfil\fi\penalty-\@M\fi\endgroup}
  \def\GPclearrowcolor{\global\let\CT@row@color\relax\global\let\CT@do@color\relax}
  \def\GPafterrow{\noalign{\ifGPstripe\global\advance\GProw by 1 \fi}%
    \ifGPstripe\ifodd\GProw\rowcolor[gray]{0.92}\fi\fi}
  \makeatother
  ```
---

# Summary

GeoPops [@hamilton2025geopops] is an open-source Python package that generates
synthetic populations for any US Census area from publicly available data.
GeoPops 1) creates a set of agents approximating US Census population
distributions, 2) assigns agents to homes, schools, workplaces, and group
quarters facilities (e.g., nursing homes) with spatial dependencies, and 3)
connects agents within these locations using network algorithms. The generated
output files include a list of agents with attributes (age, gender, income,
race/ethnicity, and more) and network structures detailing connectivity between
agents. GeoPops is designed for infectious disease modeling but can be applied
to a variety of public health issues (e.g., disaster response, education
policies) and combined with other types of connectivity networks (e.g., roads,
sewage). A GeoPops population is ideal for scenario modeling using agent-based
models and networks, and can also be used with compartmental models and contact
matrices. Users can make their own county, state, or multi-state populations.
The [GeoPopsHub](https://github.com/GeoPopsHub) on GitHub serves as a central
location for source code, tutorials, and community contributions.

# Statement of need

Synthetic populations that spatially connect individuals and capture
activity-based interactions (e.g., going to school) using networks are useful
for modeling infectious diseases, particularly when assessing the impact of
interventions on geographic and demographic subgroups. Several synthetic
population packages exist that generate agents within households for US Census
locations but do not generate activity networks. GeoPops is presently the only
one with geographically-coded school and workplace assignment and network
generation that is completely open-source and actively maintained. Synthetic
populations without pre-built networks require modelers to construct their own,
which can be data and time intensive, and limits standardization, consistency,
and comparability across models. To support both high-resolution analyses and
time-sensitive response modeling, it is essential to improve the accessibility
and usability of networked synthetic population tools. This includes providing
open-source code, clear documentation, and user-friendly interfaces and
tutorials that accommodate a range of programming experience.

# State of the field

We identified ten synthetic population generators for US Census locations
(\autoref{tab:tools}); however, few are actively maintained with open-source
code for activity assignment and network generation. They differ in geographic
and demographic granularity and method of agent/household synthesis (e.g.,
Combinatorial Optimization (CO), Iterative Proportional Fitting (IPF), or
hierarchical reconstruction (HR)). Publications for each tool show relative
accuracy for each method when comparing synthesized populations to observed
Census data. HR is faster when compared to CO and IPF but less accurate at
higher granularity. We chose CO as GeoPops' default method because it scales
well in terms of accuracy and computational requirements with increasing numbers
of target variables (i.e., individual and household characteristics). Future
iterations will allow users to select a method depending on their modeling needs
and constraints (e.g., speed vs granularity, data availability).

```{=latex}
\begingroup\footnotesize\GPwidetable\GPstripedtable
```

| | GeoPops | Geo-Synthetic-Pop | UrbanPop | SHAPE | SynthPop | UVA | SynthPops | ChiSIM | synthetic-populations |
|:-------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **Baseline population** | | | | | | | | | |
| Generates agents within households | $\checkmark$ | $\checkmark$ | $\checkmark$ | Agents only | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Open-source code | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | $\checkmark$ | | $\checkmark$ |
| Uses only publicly available data | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | | | $\checkmark$ |
| PyPI/CRAN repository | $\checkmark$ | | $\checkmark$ | $\checkmark$ | | | $\checkmark$ | | |
| Actively maintained | $\checkmark$ | | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | | $\checkmark$ |
| Downloadable pre-built state populations | All states (coming soon) | All states | Some states\* | All states | All states | Some states\* | Some states | | All states |
| **Geographic granularity** | | | | | | | | | |
| State | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| County | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Tract | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | $\checkmark$ | $\checkmark$ |
| CBG | $\checkmark$ | | $\checkmark$ | | $\checkmark$ | $\checkmark$ | | $\checkmark$ | $\checkmark$ |
| **Population synthesis years available** | | | | | | | | | |
| 2019 and earlier | $\checkmark$ | | 2016--2019 | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| 2020 and later | $\checkmark$ | 2020 only | Excluding 2020--2022 | $\checkmark$ | Upon request | Upon request | ? | $\checkmark$ | |
| **Activity networks** | | | | | | | | | |
| Generates activity networks | $\checkmark$ | $\checkmark$ | Assignment only | | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | |
| Open-source code | $\checkmark$ | $\checkmark$ | | | | | $\checkmark$ | | |
| Uses only publicly available data | $\checkmark$ | $\checkmark$ | $\checkmark$ | | ? | | $\checkmark$ | | |
| Actively maintained | $\checkmark$ | | $\checkmark$ | | $\checkmark$ | $\checkmark$ | | $\checkmark$ | |
| Downloadable network files | All states (coming soon) | All states | Assignment for some states | | All states | Some states\* | Some states\* | | |

: Capability comparison of different synthetic population generators: [GeoPops](https://github.com/GeoPopsHub) (JHU) [@hamilton2025geopops]; [Geo-Synthetic-Pop](https://github.com/njiang8/geo-synthetic-pop-usa) (U. at Buffalo) [@jiang2024]; [UrbanPop](https://github.com/likeness-pop) (Oak Ridge) [@tuccillo2023]; [SHAPE](https://github.com/evonhoene/SHAPE) (George Mason) [@vonhoene2026]; [SynthPop](https://github.com/RTIInternational/rti_synth_pop) (RTI) [@rineer2025]; Synthetic populations for regions of the World (UVA); [SynthPops](https://github.com/synthpops/synthpops) (Gates); ChiSIM (Argonne) [@macal2018]; [synthetic-populations](https://github.com/linyue-gis/synthetic-populations) (U. of Chicago) [@lin2023]. \*Others available upon request. \label{tab:tools}

```{=latex}
\endgroup
```

# Software design

GeoPops builds on its predecessor package, GREASYPOP-CO [@tulchinsky2026], and
uses the same methodology but includes several structural changes to increase
speed and usability. Data processing code has also been refactored to work with
Census data schema from 2020 and beyond, while maintaining backward
compatibility with prior years (Census boundaries and some variable codes
changed in 2020). GeoPopsHub includes a
[detailed tutorial](https://github.com/GeoPopsHub/sc_spartanburg_measles) on how
to build a population and run a measles model using the example of Spartanburg
County, SC.

## Data downloading and processing

The user specifies a geographical region, and GeoPops downloads all the
necessary data from publicly available sources (\autoref{tab:data}) through an
Application Programming Interface (API). During processing, we combine select
Census columns to avoid over granular targets in the CO step. For CO, we use 85
[target columns](https://github.com/GeoPopsHub/geopops/blob/main/target_columns.csv) that match with PUMS columns and are
relevant to disease transmission and useful in generating activity networks. We
use IPF to estimate workers by industry by residence type because Census data on
employment by CBG does not distinguish individuals living in households versus
group quarters. We also use IPF to create origin-destination and
industry-destination commute matrices by CBG from LEHD data.

```{=latex}
\begingroup\footnotesize\GPstripedtable
```

| Data source | Role in GeoPops |
|:------------------------------|:----------------------------------------------------------------------|
| [American Community Survey (ACS)](https://www.census.gov/programs-surveys/acs) | Provides Census block group (CBG) level demographic and socioeconomic distributions (e.g., age, households, income) that GeoPops matches when constructing the synthetic population. |
| [Decennial Census](https://www.census.gov/programs-surveys/decennial-census) | Used to construct group quarters compositions (institutional vs non-institutional, civilian vs military). |
| [Public Use Microdata Sample (PUMS)](https://www.census.gov/programs-surveys/acs/microdata.html) | Provides person- and household-level microdata that are reweighted and combined (via combinatorial optimization) to create realistic synthetic households and individuals. |
| [TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Provide the geographic boundaries (county, tracts, CBGs) needed to locate agents spatially and to link Census counts to geography. |
| [LEHD LODES (Longitudinal Employer--Household Dynamics)](https://lehd.ces.census.gov/data/#lodes) | Provides origin-destination commute patterns and workplace locations, used to assign workers to jobs and build the workplace contact network. |
| [County Business Patterns (CBP)](https://www.census.gov/programs-surveys/cbp.html) | Supplies employment counts by industry and geography, used to characterize and weight workplaces in the synthetic population and work network. |
| [Census Tract-to-PUMA Crosswalk](https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html) | Links Census tracts to PUMAs so that PUMS microdata (available at the PUMA level) can be aligned with tract/block-group targets. |
| [MCDC Geocorr (2018/2022)](https://mcdc.missouri.edu/applications/geocorr.html) | Provides crosswalks between tracts/CBGs and higher-level geographies (counties, CBSAs, urban-rural), used to attach regional attributes and to aggregate or filter results. |
| [NCES EDGE School Data](https://nces.ed.gov/programs/edge) | Supplies public school locations and characteristics, used to assign students and staff to schools and to build the school contact network. |

: Data sources and role in GeoPops. \label{tab:data}

```{=latex}
\endgroup
```

## Combinatorial Optimization

GeoPops uses CO with simulated annealing to generate agents within households.
CO heuristically samples real household records from PUMS data until a set is
found such that synthesized distributions of individual and household
characteristics reasonably match observed marginals from the ACS. The result is
a list of agents with attributes (age, gender, occupation, etc.) who are
assigned to households within CBG locations (\autoref{fig:households}).

![Comparison of ACS and GeoPops for number of households by household size for Spartanburg County, SC. Ninety percent confidence intervals (CI) for ACS data (blue) are calculated using margin of error values provided by the US Census [@census2019b11016]. GeoPops data (orange) include the median with minimum and maximum range from 100 population synthesis runs. The ranges for household size counts resulting from 100 GeoPops runs fall within the 90% CI ACS estimates, except for 2-person households where the GeoPops maximum overlaps with the ACS lower bound by ~50 households.\label{fig:households}](households_by_size.png)

## Assignment

Students are assigned to the closest school (Euclidean distance from the
centroid of the student's home CBG to the school's location from NCES data) with
capacity and the required grade levels with 90% probability and to the next
closest school with 10% probability. For worker assignment to workplaces, we
multiply counts of commuters by industry category in each CBG by
industry-destination commute matrices. School workers are assigned to schools by
selecting education-industry workers who commute to the same CBG as the school.
We consider five categories of group quarters facilities: institutional for ages
under 18; institutional, civilian non-institutional, and military for ages 18 to
64; and institutional for ages 65 and over. For a given CBG, the ACS provides
estimates of how many people live in each of these categories but does not
specify the exact type (e.g., nursing home or incarceration facility). In each
synthesized CBG, we create a group quarters facility of a certain category if at
least 20 residents are of the appropriate age/category status.

## Network generation

GeoPops generates network layers by default; however, users can also create
their own networks; for example, we are currently working on a between-school
network as a proxy for sports games, etc. Agents are fully connected within each
home. Within schools and workplaces, agents are connected using Stochastic Block
Modeling (SBM). For schools, the default mean degree is 12 and blocks represent
different grades. For workplaces, the default mean degree is 8 and blocks
represent income categories. For both school and workplace networks, an
assortativity coefficient of 0.9 is used, which means, for example, an average
of 90% of a student's contacts will be with other children in the same grade
(\autoref{fig:networks}A). Within group quarters facilities, agents are
connected using the small-world algorithm with a mean degree of 12 and a
rewiring probability of 0.25 (\autoref{fig:networks}B). Users can adjust network
algorithm parameters to better reflect their modeling context if needed.

![Activity network visualizations for Spartanburg County, SC. A) School network made with SBM. Agents have a mean degree of 12 and are clustered within grades within schools as a proxy for classrooms; the inset shows grade clusters within one school. The plot, created with Gephi (v0.10.1), shows some between-cluster connections. These are caused by agents who are students in one school and workers in another school; for this run, the synthetic population includes 14 agents where this is the case. B) Group quarters network where agents are connected within small-world graphs within each facility.\label{fig:networks}](networks.png)

## Running GeoPops

The user creates a parameter dictionary and passes it into the
`geopops.run_all()` function, which runs all of the construction steps
automatically (see the example below). Alternatively, the user can run each step
individually by calling functions one at a time (\autoref{tab:functions}). The
`geopops.for_starsim` module includes functions to create people and network
objects compatible with the open-source agent-based modeling software Starsim
[@kerr2024starsim]. Raw data files, interim files, and processed output are all
stored locally. This design allows users to jump quickly into agent-based
modeling with Starsim, while also enabling more detailed use by experienced
programmers. Total runtime for Spartanburg County is ~11 minutes on a 2026
MacBook Pro and household internet, and most of this is downloading data.
Runtime for the entire state of South Carolina is ~38 minutes, and for a
multi-state population of South Carolina and North Carolina ~99 minutes
(\autoref{tab:runtimes}). We will soon provide pre-built state populations for
download.

```python
import geopops

# Define parameters
pars_geopops = {
    'path': "data",                   # Folder where output files are stored
    'census_api_key': "YOUR_API_KEY", # Your Census API key
    'main_year': 2019,                # Year of data
    'geos': ["45083"],                # State or county FIPS (Spartanburg, SC)
    'commute_states': ["45", "37"],   # State FIPS of commute data (SC, NC)
    'use_pums': ["45", "37"],         # State FIPS of PUMS data (SC, NC)
}

# Generate population with .run_all()
geopops.run_all(pars=pars_geopops)
```

```{=latex}
\GPneedspace{14\baselineskip} % keep Table 3 on one page
\begingroup\footnotesize\GPstripedtable
```

| Function/module | Description |
|:-------------------------|:-----------------------------------------------------------|
| `geopops.run_all()` | Takes parameter dictionary as input. Automatically runs all the following steps sequentially (function). |
| `geopops.write_config()` | Takes parameter dictionary as input. Writes `config.json` with specified parameters (function). |
| `geopops.download_data()` | Downloads raw data files from Census API and URLs (function). |
| `geopops.process_data()` | Processes data for population generation (function). |
| `geopops.generate_pop()` | Builds the population and networks (function). |
| `geopops.for_starsim` | Exports agent and network files for Starsim (module). |

: Function and module descriptions. \label{tab:functions}

```{=latex}
\endgroup
```

| GeoPops function | Single county | Single state | Multi-state |
|:-----------------|:--------------|:-------------|:------------|
| `download_data()` | 9 min | 11 min | 18 min |
| `process_data()` | 1 min | 1 min | 4 min |
| `generate_pop()` | 1 min | 26 min | 77 min |
| **Total runtime** | **11 min** | **38 min** | **99 min** |

: Approximate runtimes for county, single-state, and multi-state populations. Single county: Spartanburg County with commute and PUMS data for SC and NC. Single state: SC with commute and PUMS data for SC, NC, and GA. Multi-state: SC and NC with commute and PUMS data for SC, NC, VA, TN, and GA. \label{tab:runtimes}

# Research impact statement

GeoPops has been presented at several research conferences, including the 2026
*Insight Net Annual Meeting* [@hamilton2026insightnet], the 2025 *Winter
Simulation Conference* [@hamilton2025wsc], *Epidemics* in 2025
[@hamilton2025epidemics], and the 2025 *MIDAS Network Annual Meeting*
[@hamilton2025midas]. The GeoPopsHub currently has 15 GitHub members. In
addition to development collaborators, researchers at other institutions are now
using GeoPops, including the Johns Hopkins University School of Public Health,
the University of Utah, and Duke University. GeoPops is also being used to teach
simulation modeling for public health decision making in an undergraduate course
for Johns Hopkins engineering students.

# AI usage disclosure

The software Cursor (v2.7--3.7) and VS Code (v1.135.0) was used in the
development of GeoPops starting with the GREASYPOP-CO source code. For example,
an AI agent was used to translate R code into Python for data downloading and
organize previous scripts into new functions. Often, the agent's first pass was
unsuccessful, in which case the chat feature was used to assist the debugging
process. AI agents were also helpful in generating comments and doc strings and
exploring raw Census data and metadata PDFs to determine where variable codes
had been changed from 2019 to 2020. Claude Code (Opus 4.5) was used to port the
Julia code for CO into Python. ChatGPT was occasionally used for more general
software development questions.

# Code and data availability

All source code and tutorials are available on
[GitHub](https://github.com/GeoPopsHub) [@hamilton2025geopops]. Soon we will
publish pre-built state populations.

# Acknowledgements

GeoPops development is a collaboration between the following institutions:

- ACCIDDA (Funding)
- Johns Hopkins University Center for Systems Science and Engineering (Design)
- Institute for Disease Modeling (Design)
- One Health Trust (Design)
- Johns Hopkins University Applied Physics Laboratory (Design)

# References
