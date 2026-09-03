# GeoPops engineering review

**Scope:** `src/geopops/` (7,207 lines, 17 modules), `tests/`, packaging. Reviewed 2026-08-28 against the goal of moving from research code to a production library.

**Verdict:** The science and the pipeline decomposition are sound — the Julia→Python port is faithful, the module boundaries (`co` → `households` → `schools` → `workplaces` → `networks` → `export`) are the right ones, and the leaf modules are already mostly plain functions. What's missing is the production layer: there is no license despite AGPL-derived code, the test suite doesn't run, there's no CI, mutable module-level globals make the pipeline non-reentrant, and several hot loops are 10–400× slower than they need to be. None of these are deep problems; most are a day or two of work each.

---

---

> **Status: implemented.** This review was carried out and then acted on in the same session. See `CHANGELOG.md` for the resulting 0.1.8 entry, and the [Implementation status](#implementation-status) section at the end for what was done, what was deliberately left, and the measured results. Sections A–G below are the original findings, kept as written.

## Priority summary

| # | Item | Severity | Effort | Section |
|---|---|---|---|---|
| 1 | No `LICENSE`; `process_data.py` is AGPL-3.0-derived | **Blocker** | 1 hr | [A1](#a1-licensing) |
| 2 | Test suite references a class that no longer exists — nothing runs | **Blocker** | 2 hr | [A2](#a2-the-test-suite-is-dead) |
| 3 | No CI | **Blocker** | 2 hr | [A3](#a3-no-ci) |
| 4 | `exit(1)` inside library code | **High** | 30 min | [A4](#a4-exit1-in-library-code) |
| 5 | Mutable module globals (`OUTPUT_DIR`, `PROCESSED_DIR`, `config`) | **High** | 1 day | [A5](#a5-mutable-module-level-globals) |
| 6 | Library writes `config.json` into its own installed package directory | **High** | 4 hr | [D1](#d1-config-is-written-into-site-packages) |
| 7 | `ForStarsim` class-level network cache is never invalidated | **High** | 1 hr | [B3](#b3-forstarsim-a-namespace-pretending-to-be-a-class) |
| 8 | Constructors that run 10-minute pipelines (`auto_run=True`) | **High** | 4 hr | [B1](#b1-classes-that-should-be-functions) |
| 9 | `anneal()` recomputes the full summary every generation | **High** | 1 hr | [C1](#c1-copy-fixes-co-annealing-10) |
| 10 | `read_workers_by_cat()` uses `.iloc[i][col]` in a triple loop | **High** | 30 min | [C2](#c2-copy-read_workers_by_cat-385) |
| 11 | Global SSL verification disabled at import time | **Medium** | 2 hr | [A6](#a6-tls-verification-disabled-globally) |
| 12 | `drawCounts()`, `find_closest()`, `households.iloc` hot loops | **Medium** | 3 hr | [C3](#c3-drawcounts-16)–[C5](#c5-per-person-iloc-in-households) |
| 13 | Default runs are silently non-reproducible (no `random_seed`) | **Medium** | 2 hr | [A7](#a7-reproducibility) |
| 14 | ~450 lines of dead code | **Medium** | 1 hr | [E1](#e1-dead-code-450-lines) |
| 15 | Zero type hints, zero logging, 175 `print()` calls | **Medium** | 2 days | [E5](#e5-printing-instead-of-logging), [E6](#e6-no-type-hints) |
| 16 | Redundant disk round-trips inside a single pipeline run | **Medium** | 1 day | [D2](#d2-round-trips-that-serve-no-purpose) |

---

## A. Blockers for a production library

### A1. Licensing

`src/geopops/process_data.py` opens with:

> Copyright 2023 Alexander Tulchinsky … Greasypop is free software: you can redistribute it and/or modify it under the terms of the **GNU Affero General Public License** … version 3 …

The repository has **no `LICENSE` file**, and `pyproject.toml` declares no `license` field. GeoPops v0.1.7 is being published to PyPI in this state. AGPL-3.0 is strongly copyleft: a derivative work that links this module must itself be AGPL-3.0-or-later. Because `process_data.py` is imported by `__init__.py`, that covers the whole distributed package.

Three options, in order of preference:

1. **Adopt AGPL-3.0-or-later for GeoPops.** Add `LICENSE`, set `license = "AGPL-3.0-or-later"` and the matching classifier in `pyproject.toml`, and note the Greasypop provenance in the README. Simplest and unambiguously correct. Note this will constrain downstream users (including Starsim integrations, which are MIT).
2. **Get relicensing permission** from Alexander Tulchinsky / One Health Trust for the derived portions, then license GeoPops under MIT/BSD to match Starsim.
3. **Rewrite `process_data.py` clean-room** — expensive and probably not worth it.

Also: `src/geopops/ipfn.py` is a vendored copy of the third-party [`ipfn`](https://pypi.org/project/ipfn/) package with its license header stripped. Either add `ipfn` as a dependency (preferred — it's actively maintained and it's 300 lines you now own) or restore the upstream copyright header and record it in a `NOTICE` file.

**Do this first.** Everything else is engineering; this is legal exposure on an already-published artifact.

### A2. The test suite is dead

```
tests/test_python_workflow.py:41:    r = geopops.RunPython()
```

`RunPython` does not exist. It was renamed to `GeneratePop` and the tests were never updated:

```
$ python -c "import geopops; print('RunPython' in dir(geopops))"
False
```

Every test in `test_python_workflow.py` fails at fixture setup. `test_julia_workflow.py` is equally stale — it calls `GPNetwork(name='homenet', beta_value=1.0)`, but the parameter has been `edge_weight` since the rename.

Three further problems with the test setup:

- **Import-time side effects.** Both files call `geopops.WriteConfig(**pars_geopops)` at module scope (`test_python_workflow.py:19`, `test_julia_workflow.py:19`). Merely *collecting* the tests writes files to the installed package directory. Move this into a fixture.
- **`tests/` is in `.gitignore`.** Three files are tracked because they predate the rule, but any new test file is silently ignored. Remove that line — it is actively hostile to growing the suite.
- **No unit tests.** The only tests are end-to-end smoke tests over a full county. There is nothing covering `lrRound`, `ranges`, `drawCounts`, `FTdist`, `urbanization_lookup`, `split_lognormal`, or the config merge logic — all pure functions that are trivial to test and easy to break.

Suggested structure:

```
tests/
  test_utils.py          # pure functions, <1s, no data
  test_config.py         # merge/override/sanitize logic
  test_co.py             # anneal on synthetic targets, seeded determinism
  test_networks.py       # SBM/small-world/complete on toy inputs
  test_regression.py     # golden-output check on tests/data (marked slow)
  test_workflow.py       # the current end-to-end smoke test (marked slow)
```

Add a **golden-output regression test**: with a fixed `random_seed`, run the pipeline on the checked-in Spartanburg fixture and assert a hash of `people.csv` and the network `.mtx` files. That single test is what lets you refactor the hot loops in section C with confidence.

### A3. No CI

There is no `.github/` directory. For a package on PyPI that means nothing verifies that a commit imports cleanly, let alone passes tests. Minimum viable:

```yaml
# .github/workflows/test.yml
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix: {python-version: ["3.11", "3.12", "3.13"]}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "${{ matrix.python-version }}"}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -m "not slow" -v
```

Add a `release.yml` that builds and publishes on tag via PyPI Trusted Publishing, and a scheduled weekly run of the slow tests (the Census/LODES endpoints change, and you want to know before a user does).

`requires-python = ">=3.11"` is currently untested against any interpreter.

### A4. `exit(1)` in library code

```
download_data.py:123:        exit(1)
download_data.py:181:        exit(1)
download_data.py:244:        exit(1)
```

A failed download terminates the host process. In a Jupyter notebook — the primary documented workflow — this kills the kernel and discards everything in memory. `exit` is also the `site` builtin, not `sys.exit`; it isn't guaranteed to exist under `python -S` or in frozen environments.

Replace with a real exception:

```python
class GeoPopsDownloadError(RuntimeError):
    """Raised when a required data file could not be downloaded."""

# ...
raise GeoPopsDownloadError(f"Download failed after {retries} attempts: {src}")
```

While you're there, define a small exception hierarchy (`GeoPopsError` → `ConfigError`, `DownloadError`, `DataError`) so callers can catch GeoPops failures specifically. Right now everything is `Exception`, `KeyError`, or `RuntimeError`, and there are 11 broad `except Exception` handlers that swallow context.

### A5. Mutable module-level globals

This is the single biggest structural problem in the codebase.

```python
# process_data.py:37-40
config = None
OUTPUT_DIR = BASE_DIR
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")

# process_data.py:1676-1680, inside ProcessData.__init__
global config, OUTPUT_DIR, PROCESSED_DIR
config = self.config
OUTPUT_DIR = self.config.get("path", self.base_dir)
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
```

`download_data.py:1236-1237` does the same with `OUTPUT_DIR`. `PROCESSED_DIR` alone is referenced 38 times across `process_data.py`; the ~1,600 lines of module-level functions there are not actually functions of their arguments — they're functions of hidden global state that a constructor happens to set.

Consequences:

- **Non-reentrant.** Two `ProcessData` instances cannot coexist. The second silently repoints the first's file paths.
- **Cannot be parallelized.** This forecloses the `sc.parallelize()` work in issue #6 for anything that touches these modules.
- **Cannot be unit-tested** without monkeypatching module globals.
- **Import-order dependent.** `read_acs()` called before any `ProcessData` exists reads from the *package* directory.

The fix is mechanical, if tedious: thread a small immutable context through the call chain.

```python
@dataclass(frozen=True)
class Paths:
    root: Path
    @property
    def processed(self) -> Path: return self.root / "processed"
    @property
    def census(self) -> Path:    return self.root / "census"
    @property
    def pums(self) -> Path:      return self.root / "pums"
    # ...

def read_acs(table, paths: Paths, geos=None): ...
```

Every function that currently reads a global takes `paths` (and `config` where needed) as an explicit parameter. Nothing about the pipeline logic changes. Do this behind the golden-output regression test from A2 and it's a safe refactor.

### A6. TLS verification disabled globally

```python
# download_data.py:18 — at import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

Importing `geopops` silently disables InsecureRequestWarning **for the entire host process**, including unrelated libraries. Combined with `verify=False` at three call sites (`download_data.py:104, 157, 222` — one of them unconditional in `try_curl_cffi`), downloaded Census/LODES data is not authenticated.

- Move `disable_warnings` out of module scope into the narrowest possible `warnings.catch_warnings()` block.
- Make the insecure fallback opt-in: `config["allow_insecure_downloads"]`, defaulting to `False`, and emit a loud `warnings.warn()` when it engages.
- `try_curl_cffi` passes `verify=False` unconditionally — that one should default to verifying and only fall back if configured to.

The browser-impersonation headers are pragmatic given how the federal data portals behave; keep them, but they don't require disabling verification.

### A7. Reproducibility

`GeneratePop._make_stage_random_seeds` (`generate_pop.py:72-88`) is a well-built piece of work — `SeedSequence.spawn(5)` gives each stage an independent, deterministic stream. But:

- **`random_seed` is absent from the shipped `config.json`.** `GeneratePop` reads `config.get("random_seed")` → `None` → every stage gets an OS-entropy RNG. So the default path is non-reproducible, and `RunAll` never exposes a seed parameter at all.
- **`process_data.py:1569` uses the unseeded legacy global RNG:**
  ```python
  sim_dist = [np.concatenate([np.random.randint(l,h,np.int64(s)) for ...
  ```
  Workplace size distributions are therefore non-reproducible regardless of what seed the caller passes.
- **`geopops_starsim.py:218`** defaults the edge-flip seed to `0` when `random_seed` is missing, so the network endpoint shuffling is deterministic while everything upstream of it isn't. Inconsistent.

Fixes: add `"random_seed": null` to the shipped config with a documented meaning; add `random_seed=` to `RunAll`; convert line 1569 to a passed-in `Generator`; and have the pipeline record the *effective* seed (including an auto-generated one when the user passes `None`) into the run's output directory so any run can be replayed.

---

## B. Classes vs functions

This addresses GeoPopsHub issue #4 directly. The codebase has 11 classes. Four earn their keep; seven are namespaces, and two of those actively cause bugs.

### The three patterns in play

**Pattern 1 — the constructor that runs the pipeline.** `WriteConfig`, `RunAll`, `DownloadData`, `ProcessData`, `QualityCheck` all end `__init__` with `if auto_run: self.run_all()`. `WriteConfig` doesn't even offer the escape hatch — constructing it always writes to disk.

This is the core problem. `DownloadData(config=cfg)` performs ten minutes of network I/O as a side effect of object construction. Constructors should construct. Beyond the principle, it produces concrete friction:

- Defaults are inconsistent (`DownloadData`/`ProcessData`/`RunAll`/`QualityCheck` default `auto_run=True`; `GeneratePop` defaults `False`; `WriteConfig` has no flag).
- `GeneratePop` carries *both* `auto_run` and a legacy `run_all` parameter that shadows the `run_all` **method** name (`generate_pop.py:28, 63-66`).
- The returned object is discarded at every call site. In `run_all.py:73-87`, `WriteConfig(...)`, `DownloadData(...)`, and `ProcessData(...)` are constructed purely for effect — three objects created and dropped.
- You can't inspect what a step *would* do before running it.

**Pattern 2 — the class as namespace.** `ForStarsim` is the clearest case (see B3). `WriteConfig` and `RunAll` are the same thing with extra steps.

**Pattern 3 — the class as counter.** `Indexer` (`utils.py:54-63`) and `DummyGenerator` (`workplaces.py:70-78`) are each a class wrapping a single integer.

### B1. Classes that should be functions

| Class | Verdict | Replacement |
|---|---|---|
| `WriteConfig` (`config.py:94`) | **Function.** `__init__` builds an overrides dict and calls `run_all()`. `get_pars()` just prints. | `write_config(path=None, **overrides) -> dict` |
| `RunAll` (`run_all.py:31`) | **Function.** Holds three attributes, all consumed once by `run_all()`. | `run_all(pars=None, config=None, seed=None, verbose=1) -> Population` |
| `QualityCheck` (`process_data.py:2078`) | **Function.** Returns a dict of diagnostics; the class adds nothing. Its `results` **property prints to stdout** (line 2148-2153) — a property with I/O side effects is a trap. | `quality_check(paths) -> dict` + `print_quality_check(results)` |
| `ForStarsim` (`geopops_starsim.py:257`) | **Module of functions.** See B3. | `to_starsim_people()`, `starsim_network()`, `SubgroupTracking` |
| `Indexer` (`utils.py:54`) | **Delete.** `d.setdefault(k, len(d) + 1)` is the whole implementation. | inline |
| `DummyGenerator` (`workplaces.py:70`) | **Closure or `itertools.count`.** | `count = itertools.count(1)` |
| `ipfn` (`ipfn.py:9`) | **Vendored third party.** | Depend on upstream `ipfn` |

`ProcessData` and `DownloadData` are the interesting middle cases. Both are *currently* namespaces around globals (A5), but both have a genuine reason to exist once the globals go: they expose per-step methods (`pull_pums_data()`, `generate_targets()`, …) so users can re-run one stage. Keep them, but:

- Make them hold real state (resolved `Paths`, validated config, an RNG) rather than mirroring it into module globals.
- Drop `auto_run`; make `run_all()` an explicit call.
- Provide thin function wrappers for the common case: `download_data(config)` / `process_data(config)`.

`GeneratePop` is the one class that unambiguously earns its keep — it holds 18 genuine pipeline intermediates between `CO()`, `SynthPop()`, and `Export()`, and users legitimately want to inspect them. Even so, I'd restructure it so the stages are free functions and the class is the *result* container:

```python
@dataclass
class Population:
    """Result of a GeoPops run; holds all pipeline intermediates."""
    config: dict
    co_results: dict | None = None
    people: dict | None = None
    # ...
    def export(self, path): ...
    def to_starsim(self): ...

def generate_pop(config, *, seed=None, verbose=1) -> Population: ...
```

That gives you `pop = geopops.generate_pop(cfg)` for the 95% case and keeps the object for inspection. `_ForStarsimGPNetwork` and `_ForStarsimSubgroupTracking` should stay classes — they subclass `ss.Network` and `ss.Analyzer` and are genuinely polymorphic.

### B2. Suggested public API

The current API mixes CamelCase classes-as-verbs with an inconsistent parameter vocabulary: `DownloadData(config=)`, `ProcessData(config_dict=)`, `GeneratePop(config_dict=)`, `ForStarsim(config_dict=)`, `RunAll(config_dict=, pars=)`. Pick one name (`config`) and use it everywhere.

```python
import geopops as gp

cfg = gp.make_config(geos=["45083"], main_year=2019, path="data")

gp.download_data(cfg)          # verbs, not nouns
gp.process_data(cfg)
pop = gp.generate_pop(cfg, seed=42)   # returns a Population
pop.export()
ppl = pop.to_starsim()

# or, all at once:
pop = gp.run(cfg, seed=42)
```

Keep the existing class names as thin deprecated aliases for one minor version so you don't break the GeoPopsHub example repos.

### B3. `ForStarsim`: a namespace pretending to be a class

`ForStarsim` (`geopops_starsim.py:257-407`) illustrates every cost of the pattern:

**Its `__init__` is entirely vestigial.** Lines 270-291 set `self.base_dir`, `self.config`, `self.path` — and no method reads any of them. `People` is a `@classmethod` that calls `cls._load_config()`; `GPNetwork` and `SubgroupTracking` are `@staticmethod`s. `main()` (line 409) constructs an instance and returns it unused.

**Class-level mutable state is a live bug.** Lines 264-268 and 204-207:

```python
class ForStarsim:
    _net_h = None; _net_s = None; _net_w = None; _net_g = None

def _ensure_networks_created(self):
    if ForStarsim._net_h is None:
        self._create_networks()
```

This cache is never invalidated. Generate one population, call `GPNetwork('homenet')`, then generate a *second* population in the same session and call `GPNetwork('homenet')` again — you silently get the first population's edges. In a notebook comparing two counties this produces wrong results with no error. As a module-level function taking an explicit path, the bug cannot occur.

**The config source is inconsistent.** `People()` honours `config_dict` and `base_dir`, but `_create_networks` hardcodes the package directory (line 212):

```python
cfg_path = os.path.join(BASE_DIR, "config.json")
```

So in `run_all.py:96-100`, `ForStarsim.People(config_dict=effective_config, base_dir=self.base_dir)` uses the caller's config while the four subsequent `GPNetwork(...)` calls read a different one from site-packages. If `base_dir` was overridden, these disagree.

Replace the whole class with module functions taking explicit paths:

```python
def to_starsim_people(pop_export_dir, *, save=True) -> ss.People: ...
def starsim_network(name, pop_export_dir, *, edge_weight=1.0, seed=None) -> ss.Network: ...
```

---

## C. Performance

All figures below are measured on this machine, not estimated. Benchmark scripts are reproducible from the snippets given.

### C1. `anneal()`: recomputes the full summary every generation — **9.6× (bit-identical)**

This is the highest-value single change in the codebase and it directly answers issue #6.

```python
# co.py:35-48 — the inner loop
while True:
    gen += 1
    cidx = rng.integers(len(c0))
    orig = c0[cidx]
    c0[cidx] = rng.integers(n_samples)
    summary = samples[c0, :].sum(axis=0, keepdims=True)   # <-- O(n_hh x n_cols), every generation
    E1 = FTdist(summary, targ)
```

Exactly **one** of `n_hh` selected households changes per generation, yet the code re-gathers all of them (allocating an `n_hh × n_cols` array) and re-sums. With `n_hh ≈ 800` and `n_cols ≈ 120` that's ~96,000 element operations per generation to reflect a change to 120 of them.

Because `samples` is `int64`, the incremental update is *exactly* equivalent — no floating-point drift:

```python
summary += samples[new]
summary -= samples[orig]
# on reject:
summary += samples[orig]
summary -= samples[new]
```

Also hoist `np.sqrt(targ + 1.0)` out of the loop — it's constant, and `FTdist` recomputes it every generation.

Measured, with `maxgens` forced to 5,000 and an identical RNG stream:

```
bit-identical indices: True | same gens: True | E0 0.2967676919127237 == 0.2967676919127237
time 0.686s -> 0.071s  (9.6x, RNG stream preserved)
```

**The output is byte-for-byte identical and the RNG call sequence is unchanged**, so this is a drop-in replacement requiring no revalidation. The speedup grows with `n_hh`, so it's largest exactly where runs are slowest — dense urban CBGs.

Batching the RNG draws (`rng.integers(..., size=4096)` outside the loop) gets a further ~1.4× to **13.8×** total, but it *does* change the RNG stream, so results shift (statistically equivalent, not identical). Take it as a second, separately-validated step.

Do this before reaching for Numba. Numba on top of the incremental version is worth maybe another 3–5× (the loop becomes ~10 µs/gen of mostly-NumPy overhead on ~120 elements), but it adds a compiled dependency. Get the algorithmic 10× free first, then measure.

**Parallelization (issue #6):** parallelize over **counties**, at `co.py:169`. Each county iteration is fully independent — it reads shared arrays and writes to its own `all_co_results[c]`. `sc.parallelize` over counties is clean. Parallelizing inside `optimize()` over CBGs would be finer-grained and better for single-county runs (the common case!), and is also safe — each `anneal()` call is independent — but you'd need to spawn per-CBG child RNG streams from a `SeedSequence` to keep results reproducible. Given a typical run is one county, **the per-CBG level is the more useful axis**; do that one.

### C2. `read_workers_by_cat()`: `.iloc[i][col]` in a triple loop — **~385×**

```python
# workplaces.py:96-100
total = sum(
    hh_samps.iloc[hh_idx[x] - 1][cat_col]
    for x in hhvec
    if x in hh_idx and pd.notna(hh_samps.iloc[hh_idx[x] - 1][cat_col])
)
```

`hh_samps.iloc[i]` constructs a fresh pandas `Series` for the whole row, then `[cat_col]` pulls one scalar out of it. The `pd.notna` guard does it **a second time**. This sits inside `for county → for cbg → for cat_code`, so for a 500-CBG county with 15 industry codes and ~1,500 households per CBG, that's roughly **22 million Series constructions**.

Measured cost of the access pattern in isolation:

```
.iloc[i][col] x2000: 28.9 ms  vs  numpy 75 us  ->  385x
```

Restructure to hoist the array conversion and do all categories at once:

```python
arr = hh_samps[cat_cols].to_numpy(dtype=float)          # once
row_of = {s: i for i, s in enumerate(hh_samps['SERIALNO'])}
for ori, hhvec in cbg_dict.items():
    rows = [row_of[x] for x in hhvec if x in row_of]
    totals = np.nansum(arr[rows], axis=0)               # all 15 categories at once
    for cat_code, t in zip(ind_codes, totals):
        workers_by_cat[cat_code][ori] = int(t)
```

This also removes the 15× redundancy of re-walking `hhvec` per category. Expect this loop to go from minutes to well under a second.

### C3. `drawCounts()`: **16×**

```python
# utils.py:104-117
for _ in range(n):
    probs = v.astype(float) / v.sum()
    idx = rng.choice(len(v), p=probs)
    v[idx] -= 1
```

This is sampling *n* items without replacement from a multiset of counts — which is exactly `Generator.multivariate_hypergeometric`, a single C call. The Python version instead does `n` calls to `rng.choice(p=...)`, each of which internally normalizes and cumsums the full probability vector.

```python
def drawCounts(v, n=1, rng=None):
    rng = rng or np.random.default_rng()
    n = min(int(n), int(v.sum()))
    if n <= 0:
        return []
    drawn = rng.multivariate_hypergeometric(v, n)
    v -= drawn
    return np.repeat(np.arange(len(v)), drawn).tolist()
```

Measured (n=200 draws over 1,500 origin bins): **6.19 ms → 0.38 ms, 16×**. This is called once per workplace and once per institution, so it runs tens of thousands of times per county.

The distribution is identical; the *order* of returned indices differs (the current version returns draw order, the replacement returns bin order). Check whether `pull_inst_workers` and `generate_workplaces` depend on that ordering — they appear to consume it as a bag, but confirm against the golden-output test. If order matters, `rng.permutation()` the result.

Two related fixes in the same call sites:

- `pull_inst_workers` (`workplaces.py:218`) recomputes `colsums = count_matrix.sum(axis=0)` — a full O(rows×cols) reduction — **once per institution**. Compute it once and decrement it as columns are drawn down.
- `generate_workplaces` (`workplaces.py:265-267`) does `count_matrix[:, col].copy()` and writes back per workplace. Slice once per destination column, mutate in place, write back once.

### C4. `find_closest()`: **64× (verified identical output)**

```python
# schools.py:40-45
for _, row in distmat.iterrows():
    dists = [(s, row[s]) for s in valid_cols if pd.notna(row[s])]
    dists.sort(key=lambda x: x[1])
    top = dists[:n]
```

`iterrows()` with per-cell `row[s]` scalar access, wrapped in an outer loop over **14 grade levels** — so the full CBG × school distance matrix is walked 14 times, cell by cell, to find 4 nearest schools per row.

```python
sub = distmat[valid_cols].to_numpy(dtype=float)
sub = np.where(np.isnan(sub), np.inf, sub)
order = np.argpartition(sub, n, axis=1)[:, :n]
rows = np.arange(len(sub))[:, None]
order = order[rows, np.argsort(sub[rows, order], axis=1)]   # sort just the top-n
```

Measured on a 600 CBG × 400 school matrix, and **verified to produce identical output**:

```
find_closest per grade: iterrows 489 ms vs vectorized 7.6 ms -> 64x
identical: True
x14 grades: 6.8s -> 0.11s
```

### C5. Per-person `.iloc` in `households.py`

```python
# households.py:203, in the innermost per-person loop
row = p_samps.iloc[r - 1]
```

Same 385× pattern as C2, executed once per **person** in the synthetic population — hundreds of thousands of times per county. Convert `p_samps` to a dict of NumPy arrays (or `itertuples()`) once, before the loop.

The two preceding lines are also row-wise `apply(axis=1)` over the whole sample frame:

```python
# households.py:175-183
p_samps['ind_code'] = p_samps[ind_colnames].apply(lambda row: first_true(row.values), axis=1)
p_samps['com_cat']  = p_samps.apply(lambda row: (row['ind_code'] + 1) if ... , axis=1)
```

`first_true` over a boolean row is `argmax`:

```python
ind = p_samps[ind_colnames].to_numpy(dtype=bool)
has_any = ind.any(axis=1)
p_samps['ind_code'] = np.where(has_any, ind.argmax(axis=1), None)
p_samps['com_cat']  = np.where(p_samps['commuter'].to_numpy(bool) & has_any,
                               ind.argmax(axis=1) + 1, None)
```

### C6. Memory: `sample_lookup` and `PersonData`

**`sample_lookup`** (`co.py:108-115`) returns *one full-length boolean mask per CBG*. For a 500-CBG county against 100,000 PUMS samples that's 500 × 100,000 = 50 MB of masks, most of them duplicates — many CBGs share a PUMA. Then `anneal` (`co.py:20`) does `all_samples[mask, :]`, materializing a fresh sub-matrix copy **per CBG**, even when the mask is identical to the previous one.

Group once, index many:

```python
groups = samp_geo.groupby(col).indices          # dict[value -> index array]
# then per CBG:  idx = groups.get(puma, EMPTY)
```

Cache the extracted `samples[idx]` sub-matrix keyed by group value. For a single county most CBGs fall in a handful of PUMAs, so this collapses ~500 gathers into ~5.

**`PersonData`** (`utils.py:19-39`) is a plain dataclass with 18 fields, instantiated once per person. Without `slots`, each instance carries a `__dict__` — roughly 600+ bytes per person, so ~300 MB for a 500k population, plus a dict of 500k tuple keys on top. `@dataclass(slots=True)` is a one-line change that cuts the per-instance overhead by more than half. The larger win would be a columnar representation (a DataFrame or a dict of arrays) instead of half a million small objects, but that's a bigger refactor — do `slots=True` now, consider columnar later.

### C7. Other hot spots

- **`generate_commute_matrices`** (`workplaces.py:346, 385`) accumulates into `sparse.lil_matrix` inside a per-origin loop. `lil` assignment is slow; collect `(row, col, val)` triplets in Python lists and build one `coo_matrix` at the end.
- **`connect_SBM`** (`networks.py:27`) builds group membership with a nested comprehension that is O(n_groups × n_keys); use a `defaultdict(list)` single pass. The zero-degree fix-up loop (lines 66-69) calls `g.degree(v)` per node — fine for small workplaces, but it's inside the per-employer loop that runs tens of thousands of times.
- **`process_data.py` writes 7 `*_test.csv` debug files** unconditionally on every run (lines 678, 737, 1223, 1286-1291). Pure I/O waste in production; gate behind a `debug` flag or delete.

---

## D. Files vs in-memory

The pipeline currently persists **everything** at every stage boundary. Some of that is right; a lot of it isn't.

### The current data flow

```
DownloadData  --> {path}/census, /pums, /geo, /work, /school     [~GB, network]
ProcessData   --> {path}/processed/*.csv         (~20 files, incl. 7 *_test.csv)
GeneratePop.CO      <-- reads processed/*.csv
  .SynthPop         --> processed/od_*.csv.gz    then immediately reads them back
  .Export           --> {path}/pop_export/*.csv, *.mtx
ForStarsim.People   <-- re-reads pop_export/*.csv, --> people_all.csv, ppl.pkl
ForStarsim.GPNetwork<-- re-reads *.mtx,           --> starsim/net_*.csv
```

**What should stay on disk:**

- **Downloaded raw data.** Expensive, remote, rate-limited, and the whole point is to cache it. Correct as is.
- **`processed/`.** This is a genuine checkpoint — it takes ~5 minutes, it's the boundary where a user might swap in their own inputs, and CO is re-run against it many times during tuning. Correct as is.
- **`pop_export/`.** The deliverable. Correct as is.

**What shouldn't:**

### D1. Config is written into site-packages

`WriteConfig.run_all()` (`config.py:150-154`) writes `config.json` **into the installed package directory** (`BASE_DIR = os.path.dirname(__file__)`). `load_config()`, `ForStarsim._load_config()`, `julia.load_config()`, `ProcessData.__init__`, and `DownloadData.__init__` all read it back from there by default.

For a library on PyPI this is the wrong model:

- Breaks on read-only installs, containers, and system-managed site-packages.
- `pip install --upgrade geopops` silently wipes the user's settings.
- Two users of the same shared install clobber each other; two concurrent runs race.
- The config in site-packages is invisible to the user's version control, so runs aren't reproducible from the repo alone.
- It required inventing the `sanitize=True` machinery (`config.py:38-52`) specifically to keep API keys from being written into the package — a workaround for a problem created by the design.

Config should live in the **user's working directory or output directory**, defaulting to `./geopops.json` or `{path}/config.json`, with the packaged file treated as a read-only template. `make_config()` should return a dict and let the caller decide whether to persist it. Note that the `.env` / `CENSUS_API_KEY` handling (`config.py:71-74`) is already the right pattern — keep that.

### D2. Round-trips that serve no purpose

**`od_*.csv.gz`.** `generate_jobs_and_workers` calls `generate_commute_matrices(data_dir)` (`workplaces.py:417`), which computes per-industry OD matrices and gzips them to `processed/od_*.csv.gz`. Four lines later, `calc_od_counts` reads all 15 of them straight back. The matrices never leave the function's own call stack. This is gzip compression + CSV serialization + parsing of a multi-megabyte sparse matrix, entirely for nothing.

Have `generate_commute_matrices` **return** the matrices, and make writing them optional (`save=True` for the checkpointing/debugging value):

```python
od_matrices = generate_commute_matrices(paths, save=save_intermediates)
origin_labels, dest_labels, od_counts = calc_od_counts(..., od_matrices=od_matrices)
```

**`GeneratePop` → `Export` → `ForStarsim`.** After `SynthPop()`, the full population is in memory (`self.people`, `self.households`, `self.adj_hh`, …). `Export()` writes it to CSV/MTX. `ForStarsim.People()` then re-reads those CSVs, does five merges, and reconstructs the same information. In `run_all.py:89-100` this happens within a single function call — the in-memory objects are alive the entire time and are simply ignored.

`ForStarsim.People` should accept either a `Population` object *or* a path:

```python
def to_starsim_people(pop_or_path): ...
```

Keep the from-disk path — loading a previously-generated population is a real use case — but don't force a serialization round-trip when the data is already in hand. Same for `_create_networks`, which re-reads the `.mtx` files that `export_networks` just wrote from `adj_hh`/`adj_sch`/`adj_wp`/`adj_gq` still in memory.

### D3. Format choices

- **CSV for adjacency data.** `adj_mat_keys.csv` and `people.csv` have one row per agent; for a large county these are hundreds of MB of text that then get re-parsed by `ForStarsim`. `pyarrow` is already a dependency — Parquet would be several times smaller and much faster to read, with dtypes preserved (which would also eliminate the `low_memory=False` and `.astype(str).replace({'na': ...})` string-coercion dance at `geopops_starsim.py:323-326`). Keep CSV as an export option for interoperability; make Parquet the internal format.
- **`.mtx` for networks.** Fine and portable, but `scipy.sparse.save_npz` is faster and smaller if the consumer is always Python.
- **`ppl.pkl`** (`geopops_starsim.py:389`) — pickle is version-fragile across Starsim releases. Fine as a convenience cache; make sure nothing in the pipeline *depends* on being able to read it back.

### D4. Intermediate results as return values

More broadly: the leaf modules (`co`, `households`, `schools`, `workplaces`, `networks`) are already written as functions that take data and return data — that's good design and it's why the pipeline is testable in principle. The friction is that several of them reach out to the filesystem for config in the middle of their work:

```python
workplaces.py:406:  config = tryJSON(os.path.join(data_dir, 'config.json'))
schools.py:62:      config = tryJSON(os.path.join(data_dir, 'config.json'))
co.py:155:          config = tryJSON(os.path.join(data_dir, 'config.json'))
```

`households.generate_people` already does this correctly — it takes `config=None` and only falls back to disk (`_resolve_config`, lines 154-158). Extend that pattern to the other three. Note that `tryJSON` **silently returns `{}` on any failure**, so a typo'd path means every tuning parameter silently reverts to its hardcoded default and the run completes with wrong numbers and no warning. That's the worst kind of failure mode for a scientific tool.

---

## E. Conciseness and clarity

### E1. Dead code (~450 lines)

| Location | Lines | What |
|---|---|---|
| `process_data.py:2198-2370` | ~172 | `generate_test_targets`, `gen_samp_test_cols`, `test_cols` — test scaffolding shipped in the package, called by nothing |
| `download_data.py:1438-1585` | ~147 | `DownloadData.pipeline()` — prints a hardcoded description of what the other methods do; guaranteed to drift out of sync. This is documentation, not code |
| `networks.py:247-335` | 89 | `generate_location_matrices` — never called |
| `julia.py` | 123 | Legacy Julia path. `main()` (line 117) has its only real statement commented out |
| `process_data.py`, various | — | ~25 commented-out `# print(...)` and `# ....to_csv(...)` lines (223, 232, 238, 246, 490, 496, 507, 591, 632, 707, 735, …) |

`julia.py` deserves a decision rather than deletion by default: it's the reference implementation the Python port was validated against. Either keep it deliberately (document it as "reference only", exclude from `__all__`) or drop it and rely on the git history plus a golden-output test. Shipping it in `__all__` as a peer of `GeneratePop` implies it's supported, and `RunJulia.__init__` raises `ValueError` for any user who hasn't configured a Julia environment.

### E2. Duplicated helpers

- **`tryJSON`** is defined twice: `utils.py:11` (silently returns `{}`) and `process_data.py:25` (prints a warning). Different behaviour, same name.
- **`lrRound`** is defined twice with *different implementations*: `utils.py:66` (NumPy) and `process_data.py:44` (pandas Series, mutating in place via label indexing). Two largest-remainder rounders that can disagree on ties is a latent scientific bug.
- **Three near-identical download functions** — `try_download` (line 77), `try_curl_cffi` (127), `try_download_text` (185) — ~180 lines implementing the same retry/SSL-fallback loop three times. One function with `backend=` and `mode=` parameters replaces all three.

### E3. Repeated blocks that want a loop

**`co.py:176-215`** — the four optimization passes (PUMA → county → CBSA → urbanization) are the same ten lines copy-pasted four times, differing only in the column name and lookup dict. ~55 lines becomes ~15:

```python
LEVELS = [("PUMA", "st_puma", cbg_puma, params),
          ("county", "county", cbg_county, params),
          ("CBSA",   "cbsa",   cbg_cbsa,   params),
          ("urbanization", "U", cbg_urban, params_slow)]

for label, col, lookup, p in LEVELS:
    rerun = [i for i, r in enumerate(x) if r[2] > c_val]
    if not rerun:
        continue
    masks = sample_lookup(samp_geo, col, [lookup[geos[i]] for i in rerun])
    reoptimize(x, rerun, samples, masks, targs, n_hhs, p, rng)
    _report(label, x, c_val)
```

Note this also fixes a latent bug: the current code prints `"Optimizing 0 CBG(s) at county level"` and *then* checks `if rerun:` — so at the CBSA and urbanization stages it announces work it doesn't do.

**`geopops_starsim.py:327-336`** — ten near-identical `.loc[]` assignments to build age groups:

```python
ppl_df['agegroup'] = np.clip(ppl_df['age'] // 10, 0, 9)
```

**`export.py`** — nine blocks of `sorted([...]) → pd.DataFrame(..., columns=[...]) → to_csv → _log_export`. A small helper collapses each to one line:

```python
def _write(rows, columns, name, key=None):
    pd.DataFrame(sorted(rows, key=key) if key else rows, columns=columns).to_csv(export_dir / name, index=False)
    _log_export(verbose, f"-- {rel}/{name}")
```

**`run_all.py:96-100`** — four `ForStarsim.GPNetwork(name=..., edge_weight=1.0)` calls; `for name in ('homenet', 'schoolnet', 'worknet', 'gqnet')`.

### E4. Convoluted logic

**`co.py:130-132`:**

```python
enough = [samp_masks[i].sum() > (n_hhs[rerun[j]] // 2) for j, i in enumerate(range(len(samp_masks)))]
valid = [rerun[j] for j, ok in enumerate(enough) if ok]
valid_mask_idx = [j for j, ok in enumerate(enough) if ok]
```

`for j, i in enumerate(range(len(samp_masks)))` means `j == i` unconditionally, and `valid_mask_idx` is just the surviving `j` values. The whole thing is:

```python
for j, ri in enumerate(rerun):
    if samp_masks[j].sum() <= n_hhs[ri] // 2:
        continue
    r = anneal(samples, samp_masks[j], targs[ri:ri+1], n_hhs[ri], params, rng)
    if r[2] < x[ri][2]:
        x[ri] = r
```

**`co.py:121`:** `zip(samp_masks, range(len(targs)), n_hhs)` → `zip(samp_masks, targs, n_hhs)`.

**`co.py:170-171`:** `cmask = [co == c for co in county_of]` then `idxs = [i for i, m in enumerate(cmask) if m]` → `idxs = [i for i, co in enumerate(county_of) if co == c]`. (The loop variable `co` also shadows the imported `co` module.)

**`generate_pop.py:100-176`** — five `_log_*_summary` methods, ~75 lines, are pure logging and make up 30% of the class. Move them to a `summary.py` module, or better, have the stages return small summary dicts and log them uniformly.

### E5. Printing instead of logging

175 `print()` calls across the package (80 in `download_data.py`, 48 in `process_data.py`). The `verbose` flag is threaded by hand through every function that needs it, with inconsistent semantics — `DownloadData` documents `verbose` as `1`/`0`, `export_synthpop` defaults it to `True`, `GeneratePop` uses truthiness.

Use `logging` with a package logger:

```python
logger = logging.getLogger("geopops")
```

Then `verbose=` on the public entry points just sets a level, users can redirect or silence output through standard mechanisms, and library code stops writing to stdout unconditionally. `ipfn.py` prints 7 times from inside a numerical inner loop.

`process_data.py:1319` has a live example of the cost: `x_c = test_c.sub(ref_c).apply(abs).apply(lambda s: (s > 5.0).any(), axis=1)` computes a diagnostic whether or not anyone will look at it.

### E6. No type hints

Zero annotated function signatures in 7,207 lines. For a library whose core data structures are undocumented nested tuples — `people` is `dict[tuple[int,int,int], PersonData]`, `company_workers` is `dict[tuple[int,int,str], list[tuple[int,int,int,int]]]`, worker tuples are variously sliced `[:3]` and indexed `[3]` — this is the difference between an API a new contributor can use and one they have to reverse-engineer.

Start with the public entry points and the inter-module boundaries (the return signature of `generate_networks` is an 8-tuple; `generate_jobs_and_workers` returns a 5-tuple). Named tuples or small dataclasses for the key types would help as much as the annotations:

```python
class PersonKey(NamedTuple):
    p_id: int
    hh_id: int
    cbg_id: int
```

Then `w[:3]` becomes `w.person` and `k[2]` becomes `k.cbg_id` throughout. Add `mypy` (or `ty`) to CI in non-strict mode and tighten over time.

### E7. Fragile config→code coupling

`households.py:204-207` passes `additional_traits` from config into `PersonData(**trait_kwargs)`, but `PersonData` (`utils.py:19-39`) has a **hardcoded** field list. The shipped config's 11 `additional_traits` happen to match exactly. Add a twelfth trait — the natural thing to do after reading issue #2's discussion of race/ethnicity categories — and the pipeline dies with an opaque `TypeError` deep in the person loop.

Either derive the traits dynamically, or validate up front with a clear message:

```python
known = {f.name for f in dataclasses.fields(PersonData)}
unknown = set(additional_traits) - known
if unknown:
    raise ConfigError(f"additional_traits not supported by PersonData: {sorted(unknown)}")
```

Given issue #2's caveat — that these traits are carried through from PUMS but are *not* used in the CO step, so their distributions aren't matched to ACS — this is worth surfacing in code as well as in the issue: a warning when a trait is requested that CO doesn't target.

### E8. Small correctness items

- **`generate_pop.py:96-98`** — `_county_from_cbg_idx` reads `self._cbg_by_idx`, which is only created inside `SynthPop()` (line 199) and isn't declared in `__init__` alongside the other 18 state attributes. Calling `Export()` on a partially-run instance gives `AttributeError` rather than the clear `RuntimeError` the other stages raise.
- **`geopops_starsim.py:386-387`** — `sim = ss.Sim(people=ppl).init()` then `_ = sim  # keep side-effect parity`. A comment admitting the code depends on an unnamed side effect. Work out what `init()` is actually doing to `ppl` and call that directly, or document it.
- **`config.py:55-59`** — `compute_decennial_year` catches bare `Exception` and returns `2010` for any unparseable input, so `main_year="twenty-nineteen"` yields a plausible-looking config that produces wrong data.
- **`utils.py:66-75`** — `lrRound` handles `vrem > 0` but not `vrem < 0`, which can occur with negative inputs. Probably unreachable given the call sites, but it should assert rather than silently under-round.
- **`config.py:50`** — `os.makedirs(os.path.dirname(cfg_path), exist_ok=True)` raises `FileNotFoundError` when `cfg_path` is a bare filename (`dirname` returns `""`).

---

## F. Packaging and project metadata

This covers issue #3.

`pyproject.toml` is missing: `authors`, `maintainers`, `license`, `keywords`, `classifiers`, `[project.urls]` (Homepage / Documentation / Repository / Issues / Changelog). PyPI currently shows a package with no author, no license, and no links. Compare `starsim`'s for a template.

Also:

- **No `LICENSE`** (see A1).
- **No `CHANGELOG.md`.** Version is at 0.1.7 with no record of what changed. Given that the class rename (`RunPython` → `GeneratePop`) and the parameter rename (`beta_value` → `edge_weight`) both broke callers — the tests among them — this matters now.
- **No `CONTRIBUTING.md`** despite the README's "Get involved" section actively soliciting contributions.
- **Unpinned dependencies.** `geopandas`, `networkx`, `numpy`, `sciris`, `scipy`, `shapely` have no lower bounds. `numpy` in particular matters: `Generator.multivariate_hypergeometric` (recommended in C3) needs ≥1.18, and `SeedSequence.spawn` needs ≥1.17.
- **`build/` is committed** and contains a stale copy of the package (`build/lib/geopops/pyjulia/`, `census.py`) that no longer matches `src/`. It's gitignored but present in the working tree; it will confuse grep-based navigation and any tooling that walks the tree. Delete it.
- **`README.md.orig`** is an untracked leftover — remove or restore it.
- **`.gitignore` contains `tests/`** (see A2) and `tutorials/`, which means the tutorials the README points to can't live in this repo.
- **No `py.typed`** marker (add once E6 is underway).
- **No linter config.** Add `ruff` with a modest rule set and run it in CI; it will catch the shadowed `co` variable, the unused imports, and the bare excepts automatically.

---

## G. Suggested sequencing

**Week 1 — stop the bleeding**

1. Add `LICENSE` + `pyproject.toml` license metadata (A1). Decide on the `ipfn` vendoring.
2. Fix the tests so they run at all: `RunPython` → `GeneratePop`, `beta_value` → `edge_weight`, move the `WriteConfig` call into a fixture, un-ignore `tests/` (A2).
3. Add the golden-output regression test on the Spartanburg fixture with a fixed seed. **This is the enabler for everything downstream.**
4. Add CI (A3).
5. Replace `exit(1)` with exceptions (A4).

**Week 2 — free performance**

6. Incremental `anneal()` summary (C1). Bit-identical, so the golden test proves it immediately. ~10× on the dominant cost.
7. Vectorize `read_workers_by_cat` (C2), `find_closest` (C4), and the `households` per-person `.iloc` (C5).
8. `drawCounts` → `multivariate_hypergeometric` (C3), plus the `colsums` hoist in `pull_inst_workers`.
9. Delete the dead code (E1) and the `*_test.csv` writes.

At this point the runtime should be dramatically better and you'll know exactly where the remaining time goes — profile again before considering Numba or parallelism.

**Weeks 3–4 — structure**

10. Kill the module globals; introduce `Paths` and thread config explicitly (A5). Biggest single refactor, safest with the golden test in place.
11. Move config out of site-packages (D1).
12. Convert the namespace classes to functions (B1, B3), keeping deprecated aliases. Fix the `ForStarsim` cache bug on the way.
13. Remove the `od_*.csv.gz` round-trip; let `ForStarsim` accept in-memory objects (D2).
14. Add `random_seed` to the shipped config and to `RunAll`; seed `process_data.py:1569` (A7).

**Ongoing**

15. Type hints from the public API inward; `NamedTuple` for `PersonKey` and the worker tuples (E6).
16. `logging` instead of `print` (E5).
17. Consolidate the duplicated helpers and the three download functions (E2).
18. Parallelize CO over CBGs with reproducible per-CBG seeds (C1).
19. Docs (issue #5) and project metadata (issue #3 / section F).

---

## Appendix: measurements

All benchmarks run on this machine, 2026-08-28. Sizes chosen to approximate a mid-sized county.

| Change | Before | After | Speedup | Output |
|---|---|---|---|---|
| `anneal` incremental summary (5k gens, n_hh=800, 120 cols) | 0.686 s | 0.071 s | **9.6×** | bit-identical, RNG stream preserved |
| `anneal` + batched RNG draws (20k gens) | 2.74 s | 0.20 s | **13.8×** | statistically equivalent |
| `.iloc[i][col]` → NumPy indexing (2,000 lookups) | 28.9 ms | 0.075 ms | **385×** | identical |
| `drawCounts` → `multivariate_hypergeometric` (n=200, 1,500 bins) | 6.19 ms | 0.38 ms | **16×** | same distribution, different order |
| `find_closest` per grade (600 CBG × 400 schools) | 489 ms | 7.6 ms | **64×** | verified identical |
| `find_closest` × 14 grades | 6.8 s | 0.11 s | **64×** | verified identical |


---

## Implementation status

Everything below was done in this session and verified against the Spartanburg County, SC fixture (195 CBGs, ~297k people, ~357k agents including dummies). The work is uncommitted in the working tree.

### Done

| Item | What changed |
|---|---|
| [A1](#a1-licensing) | Added `LICENSE` (AGPL-3.0-or-later, verbatim FSF text) and `NOTICE` recording the GREASYPOP-CO provenance and restoring the `ipfn` MIT attribution. Declared `license` in `pyproject.toml`; both files ship in the wheel. |
| [A2](#a2-the-test-suite-is-dead) | Replaced the two dead test files with 96 tests across `test_utils`, `test_config`, `test_co`, `test_networks`, `test_sources`, `test_workflow`, `test_regression`. Added `conftest.py` fixtures; import-time side effects gone; `tests/` un-ignored (only `tests/data/` is ignored now). |
| [A3](#a3-no-ci) | `.github/workflows/test.yml` (pytest on 3.11/3.12/3.13 + ruff) and `release.yml` (build, `twine check`, PyPI Trusted Publishing). |
| [A4](#a4-exit1-in-library-code) | All three `exit(1)` calls replaced by `DownloadError`. Added a `GeoPopsError` hierarchy. |
| [A6](#a6-tls-verification-disabled-globally) | Removed the import-time `urllib3.disable_warnings()`. The unverified-TLS fallback is now opt-in via `allow_insecure_downloads`, warns loudly, and scopes warning suppression to the single request. |
| [A7](#a7-reproducibility) | `random_seed` added to the shipped config; `run()`/`generate_pop()` take `seed=`; `generate_work_sizes` no longer uses numpy's unseeded global RNG; unseeded runs warn. |
| [B1](#b1-classes-that-should-be-functions), [B2](#b2-suggested-public-api), [B3](#b3-forstarsim-a-namespace-pretending-to-be-a-class) | `WriteConfig`, `RunAll`, `ForStarsim`, `RunJulia` removed in favour of `make_config`, `run`, `to_starsim_people`/`starsim_network`/`starsim_networks`. Fixed the never-invalidated `ForStarsim` network cache and the config mismatch between `People()` and `GPNetwork()`. `DownloadData`/`ProcessData`/`GeneratePop` kept as step objects. |
| [C1](#c1-copy-fixes-co-annealing-10)–[C7](#c7-other-hot-spots) | All the hot-path work: incremental annealing summary, hoisted `sqrt(targ+1)`, cached sample sub-pools, vectorized `read_workers_by_cat` / `find_closest` / `generate_people`, hypergeometric `drawCounts`, incremental column sums in `pull_inst_workers`, COO instead of LIL in `generate_commute_matrices`, `__slots__` + shared trait schema on `PersonData`. |
| [D1](#d1-config-is-written-into-site-packages) | Config no longer written into site-packages. `make_config()` returns a dict; `save=True` writes into the run directory. The packaged `config.json` is a read-only template. |
| [D2](#d2-round-trips-that-serve-no-purpose) | `generate_commute_matrices` returns its matrices instead of gzipping them and reading them straight back. `to_starsim_people`/`starsim_networks` take an explicit `pop_export_dir`, and `GeneratePop.pop_export_dir` supplies it. |
| [E1](#e1-dead-code-450-lines) | Removed `generate_location_matrices`, the `generate_test_targets`/`gen_samp_test_cols`/`test_cols` scaffolding, `DownloadData.pipeline()`, `julia.py`, the 11 `julia/*.jl` files, and the stale empty `pyjulia/` directory. |
| [E2](#e2-duplicated-helpers) | Three near-identical download functions collapsed into one `download(src, dst, backend=, mode=)`. |
| [E3](#e3-repeated-blocks-that-want-a-loop), [E4](#e4-convoluted-logic) | CO's four copy-pasted passes became a loop over levels (also fixing the misleading "Optimizing 0 CBGs" log); `reoptimize`'s `enumerate(range(len(...)))` untangled; ten `agegroup` assignments became one `np.clip`. |
| [E7](#e7-fragile-configcode-coupling) | Traits are config-driven via `TraitSchema` and reached by attribute as before. **This was a live bug, not a latent one:** the checked-in fixture config lists `white_non_hispanic`, so the fixture could not run against 0.1.7 at all — `TypeError: unexpected keyword argument`. |
| [E8](#e8-small-correctness-items) | `compute_decennial_year` raises instead of defaulting to 2010; `save_config` handles a bare filename; `QualityCheck.results` no longer prints; `_cbg_by_idx` declared up front; exception chaining and loop-invariant hoisting fixed. |
| [F](#f-packaging-and-project-metadata) | Full `pyproject.toml` metadata (authors, license, keywords, classifiers, URLs), dependency lower bounds, ruff + pytest config, `CHANGELOG.md`, updated `README.md`, cleaned `.gitignore`, removed the Julia entry from `.env.example`, deleted the stale `build/` tree. `ruff check src/ tests/` passes clean. |

One thing surfaced during implementation that wasn't in the original review: exporting `download_data`/`process_data`/`generate_pop` as functions shadowed the modules of the same name. The modules were renamed to `sources.py`, `census.py`, `population.py`, `starsim_bridge.py`, and `pipeline.py`.

### Measured results

Spartanburg County, SC (195 CBGs, 292k people); `CO_maxgens=20000`; seed 42.

| Stage | Before | After | Speedup | Output |
|---|---|---|---|---|
| CO (`process_counties`) | 2192.9 s | 41.1 s | **53.3×** | **identical** |
| SynthPop | 200.0 s | 18.3 s | **10.9×** | see below |
| Export | 4.0 s | 4.1 s | — | — |
| **Total** | **~36.6 min** | **~63 s** | **~35×** | |

Each rewritten component was checked against the original implementation on the real fixture, not just benchmarked:

| Component | Before | After | Speedup | Equivalence check |
|---|---|---|---|---|
| `co.process_counties`, 195 CBGs | 2192.9 s | 41.1 s | 53.3× | **identical** — `co_results` and `co_scores` compared whole |
| `co.anneal`, 6 seeded trials incl. empty pool and early exit | 0.686 s | 0.071 s | 9.6× | **bit-identical**, RNG call sequence preserved |
| `households.generate_people`, 292,039 people | 46.9 s | 6.3 s | 7.4× | **identical** — every core field, every trait, all households, all GQs, and `gq_summary` |
| `schools.find_closest`, 14 grades | 0.36 s | 0.035 s | 10× | **identical** across 2,730 CBG/grade entries |
| `.iloc[i][col]` → NumPy (2,000 lookups) | 28.9 ms | 0.075 ms | 385× | identical |
| `drawCounts` (n=200, 1,500 bins) | 6.19 ms | 0.38 ms | 16× | same distribution, different draw order |
| `PersonData` memory | ~1,515 B/person | ~240 B/person | 6.3× smaller | — |

The 53× on CO exceeds the 9.6× that `anneal` alone gives, because caching the candidate sub-pools also removes a per-CBG re-extraction of the same PUMA sub-matrix.

`census.py` (the former `process_data.py`) was validated by running it end to end: it completes in 149 s and **24 of its 27 outputs are byte-identical** to the reference. Of the three that differ, `work_sizes.csv` is by design (it is now seeded rather than using numpy's unseeded global RNG), and `p_samples.csv`/`samp_geo.csv` differ only in `st_puma`/`PUMA` zero-padding — a **pre-existing** discrepancy, since `HEAD` already applies `zfill(5)` and the checked-in fixture predates it. See "Two things worth a decision" below.

### Output equivalence

Running the full pipeline before and after the workplace changes, same seed:

- **Bit-identical**: `people.csv`, `hh.csv`, `cbg_idxs.csv`, `sch_students.csv`, `gqs.csv`, `gq_residents.csv`, `adj_mat_keys.csv`, `adj_dummy_keys.csv`, `adj_out_workers.csv`, `adj_upper_triang_hh.mtx`.
- **Changed, as designed**: `company_workers.csv`, `sch_workers.csv`, `gq_workers.csv`, and the school/workplace/GQ/non-household network matrices — everything downstream of `drawCounts`, which now consumes the RNG stream differently. The distribution is unchanged; the specific assignment is not. This is the one behavioural change in the release and it is called out in `CHANGELOG.md`.

`tests/golden_hashes.json` pins the post-change output, and `test_same_seed_gives_the_same_population` asserts run-to-run reproducibility independently of that baseline.

### Deliberately not done

- **[A5](#a5-mutable-module-level-globals) — the module-global refactor in `census.py`/`sources.py`.** `PROCESSED_DIR` alone is referenced 38 times across 2,200 lines. It is the right next change and it is what unblocks parallelizing CO ([C1](#c1-copy-fixes-co-annealing-10)), but it is a large mechanical edit to the least-tested module, and the golden test that would make it safe only exists as of this session. Doing it now would have put the riskiest change on the least evidence. It is recorded under "Known issues" in the changelog.
- **Parallelizing CO across CBGs.** CO went from 37 minutes to 41 seconds without it, so the payoff is much smaller than it was, and it depends on A5.
- **[E5](#e5-printing-instead-of-logging) `logging` and [E6](#e6-no-type-hints) type hints.** Both are broad, low-risk, and mostly mechanical; neither blocks anything. Worth doing incrementally from the public API inward.
- **[D3](#d3-format-choices) Parquet internally.** A real win for large counties, but it changes the on-disk contract for downstream consumers and deserves its own release.
- **The seven `*_test.csv` debug writes** in `census.py`, which live in the middle of the module-global code A5 covers.
- **Documentation ([issue #5](https://github.com/GeoPopsHub/geopops/issues/5)).** The README and changelog are updated, but a built docs site is a separate piece of work.

### Two things worth a decision

1. **The license.** AGPL-3.0-or-later is the only option available without third-party permission, given that `census.py` is derived from GREASYPOP-CO. It is also viral, which will constrain downstream users — including Starsim integrations, which are MIT. If a permissive license matters, that is a conversation to have with the GREASYPOP-CO copyright holders, and it should happen before more releases go out.
2. **The fixture data is stale, in two independent ways.** `tests/data/processed/p_samples.csv` carries the pre-[issue #2](https://github.com/GeoPopsHub/geopops/issues/2) trait columns (`white_non_hispanic` rather than the eight PUMS categories) — which is why the fixture could not run against 0.1.7 at all. Separately, its `st_puma` values are unpadded (`45501`) while the current code emits `zfill(5)`-normalized ones (`4500101`); regenerating `processed/` changes those two files. Neither is caused by this session's work, but together they mean the fixture no longer represents what the pipeline produces, which limits what the golden test proves. It is also large and gitignored, so it is not reproducible from a fresh clone — **a small committed fixture (one or two CBGs) would let CI run the end-to-end and regression tests at all**, which is currently the biggest remaining gap in the test story.
