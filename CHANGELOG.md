# Changelog

All notable changes to GeoPops are documented here. This project follows [semantic versioning](https://semver.org/).

## [0.1.8] — unreleased

This release is an engineering pass over the whole package: licensing and packaging, a working test suite and CI, a single functional API, and substantial performance work. **It contains breaking API changes** (see below); the science and the output file formats are unchanged except where noted.

### Licensing

- **Added `LICENSE` (AGPL-3.0-or-later) and `NOTICE`.** GeoPops was previously distributed with no license despite `src/geopops/process_data.py` being derived from [GREASYPOP-CO](https://github.com/CDDEP-DC/GREASYPOP-CO) (Copyright 2023 Alexander Tulchinsky), which is AGPL-3.0-or-later. `NOTICE` records the GREASYPOP-CO provenance and restores the MIT license attribution for the vendored `ipfn` module. Relicensing GeoPops permissively would require permission from the GREASYPOP-CO copyright holders.

### Breaking changes

- **Removed the "class as verb" API.** `WriteConfig`, `RunAll`, `ForStarsim`, and `RunJulia` are gone, along with `geopops/julia.py`. Constructors no longer run pipelines as a side effect.

  | Before | Now |
  |---|---|
  | `geopops.WriteConfig(**pars)` | `geopops.make_config(**pars)` |
  | `geopops.RunAll(pars=pars)` | `geopops.run(cfg)` |
  | `geopops.DownloadData(config=cfg)` | `geopops.download_data(cfg)` |
  | `geopops.ProcessData(config_dict=cfg)` | `geopops.process_data(cfg)` |
  | `geopops.GeneratePop(config_dict=cfg, auto_run=True)` | `geopops.generate_pop(cfg, seed=...)` |
  | `geopops.ForStarsim.People()` | `geopops.to_starsim_people(pop.pop_export_dir)` |
  | `geopops.ForStarsim.GPNetwork(name=...)` | `geopops.starsim_network(name, pop.pop_export_dir)` |
  | `geopops.ForStarsim.SubgroupTracking(...)` | `geopops.SubgroupTracking(...)` |
  | `geopops.RunJulia()` | removed; use `geopops.generate_pop` |

  `DownloadData`, `ProcessData`, and `GeneratePop` remain as objects for running or inspecting individual stages.

- **Config is no longer written into the installed package.** `make_config()` returns a dict; `save=True` writes `config.json` into the run's own output directory. Previously `WriteConfig` wrote into `site-packages`, which broke read-only installs, was wiped by `pip install --upgrade`, and made concurrent runs race. The packaged `config.json` is now a read-only template.
- **`PersonData` traits are config-driven.** Traits named in `config["additional_traits"]` are carried in a `TraitSchema`-positioned tuple and reached by attribute (`person.hispanic`) as before. Adding a trait no longer requires editing the dataclass, which previously raised `TypeError: unexpected keyword argument`.
- **`people.csv` column order changed.** Trait columns now follow `commuter_workplace_category` and are whatever the config asked for, instead of a hardcoded list. Read columns by name, not position.
- **Downloads raise instead of exiting.** `try_download`, `try_curl_cffi`, and `try_download_text` are replaced by a single `download(src, dst, backend=..., mode=...)` that raises `DownloadError`. The old functions called `exit(1)`, which killed the host process — including Jupyter kernels.
- **Workplace assignment results differ for a given seed.** `drawCounts` now uses `Generator.multivariate_hypergeometric`, which draws from the same distribution but consumes the RNG stream differently. Population, household, school, group-quarters, and household-network outputs are bit-identical to 0.1.7; the workplace, school-worker, GQ-worker, and non-household network outputs are statistically equivalent but not identical.

### Added

- **Test suite and CI.** 78 unit tests covering config, utilities, CO, and networks, plus slow end-to-end and golden-output regression tests. The previous test files referenced `geopops.RunPython`, a class that no longer existed, so nothing ran. GitHub Actions workflows run tests on Python 3.11/3.12/3.13 and lint with ruff.
- **Exception hierarchy**: `GeoPopsError` and its subclasses `ConfigError`, `DownloadError`, `DataError`, `PipelineStateError`.
- **`validate_config()`**, run automatically by `make_config()`. Unknown override keys are now an error rather than silently ignored, an unparseable `main_year` raises instead of defaulting to a 2010 vintage, and an unseeded run warns.
- **`starsim_networks()`** builds all four layers, reading each matrix file once.
- **`GeneratePop.pop_export_dir`**, so downstream steps need not reconstruct the path.
- **`allow_insecure_downloads`** config flag (default `False`).
- **Project metadata**: authors, license, keywords, classifiers, and URLs in `pyproject.toml`; dependency lower bounds; ruff and pytest configuration.

### Fixed

- **Starsim networks were cached in class-level state and never invalidated.** Generating a second population in the same session silently reused the first population's edges. The new functions hold no cross-call state.
- **`ForStarsim.GPNetwork` read a different config from `ForStarsim.People`**, hardcoding the packaged `config.json` while `People()` honoured `config_dict`/`base_dir`.
- **Workplace size distributions were not reproducible**: `generate_work_sizes` used numpy's unseeded global RNG. It now takes `random_seed`, threaded from `config["random_seed"]`.
- **`QualityCheck.results` printed to stdout** as a side effect of attribute access.
- **`compute_decennial_year` silently returned 2010** for any unparseable `main_year`.
- **`save_config` crashed** when given a bare filename with no directory component.
- Loop-invariant constants were rebound inside a loop body in `pull_census_data`; exceptions raised inside `except` blocks were not chained.

### Performance

Measured on the Spartanburg County, SC fixture (195 CBGs, ~300k people), `CO_maxgens=20000`:

| Stage | Before | After | Speedup |
|---|---|---|---|
| CO (`process_counties`) | 2192.9 s | 41.1 s | **53×** (identical output) |
| SynthPop | 200.0 s | 18.3 s | **11×** |
| Export | 4.0 s | 4.1 s | — |
| **Total** | **~40 min** | **~64 s** | **~37×** |

The changes behind this:

- `anneal()` updates its running column-sum incrementally instead of re-gathering and re-summing every selected household each generation. Because the sample matrix is integer, this is exact — verified bit-identical with an unchanged RNG call sequence.
- `sqrt(target + 1)` is hoisted out of the annealing loop.
- Candidate sample sub-pools are cached by lookup key, so CBGs sharing a PUMA no longer each re-extract the same sub-matrix.
- `read_workers_by_cat` sums a NumPy view column-wise instead of `.iloc[i][col]` per household per industry (~385× on the access pattern alone).
- `find_closest` uses `argpartition` instead of `iterrows` with per-cell access (~10–64×, verified identical output).
- `drawCounts` uses one `multivariate_hypergeometric` call instead of a loop of `rng.choice(p=...)` (~16×).
- `generate_people` precomputes person attributes column-wise instead of `.iloc`/`.apply(axis=1)` per person.
- `pull_inst_workers` maintains column sums incrementally rather than recomputing the full reduction per institution.
- `generate_commute_matrices` accumulates COO triplets instead of assigning into a `lil_matrix`, and **returns** its matrices — they were previously gzipped to `od_*.csv.gz` and read straight back four lines later.
- `PersonData` uses `__slots__` with a shared trait schema: ~240 bytes per person, down from ~1.5 kB (~600 MB saved on a 500k population).

### Removed

- ~450 lines of dead code: `generate_location_matrices` (never called), the `generate_test_targets`/`gen_samp_test_cols`/`test_cols` scaffolding shipped in `process_data.py`, `DownloadData.pipeline()` (a hardcoded prose description of the other methods), and `geopops/julia.py`.
- Two of the three near-identical download functions.
- `urllib3.disable_warnings()` at import time, which disabled TLS warnings process-wide for every library in the host process.

### Known issues

- `process_data.py` and `download_data.py` still set module-level globals (`config`, `OUTPUT_DIR`, `PROCESSED_DIR`) from their step objects, so those two stages are not reentrant or safely parallelizable. Threading an explicit path/config context through them is the next structural change.
- `process_data.py` writes seven `*_test.csv` debug files on every run.
- CO is not yet parallelized across CBGs. (Much less pressing now that it runs in 41 s rather than 37 minutes, and it depends on the module-global refactor above.)
- There are no type hints, and logging still goes through `print`.
- The `tests/data` fixture is stale and gitignored, so CI cannot run the end-to-end or golden-output tests. Its `processed/` files predate both the issue #2 trait change and the `st_puma` zero-padding normalization. A small committed fixture (one or two CBGs) would fix this.

## [0.1.7] — 2026-08

- Replaced the Julia implementation with pure Python.
- Reworked race/ethnicity traits: removed the non-mutually-exclusive `race_ethnicity` variable in favour of the eight PUMS categories. See [issue #2](https://github.com/GeoPopsHub/geopops/issues/2). Note that these traits are carried through from PUMS but are *not* targeted by the CO step.
- Sanitized config handling so API keys are not written into the packaged template.
