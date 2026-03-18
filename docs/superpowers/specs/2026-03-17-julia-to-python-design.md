# Julia-to-Python Conversion Design

## Motivation

Remove the Julia dependency so users don't need a separate Julia environment to run GeoPops. Secondary goals: improve maintainability (single language) and enable future Python-native performance optimization (e.g., Numba).

## Approach

File-by-file translation: each Julia file maps to a corresponding Python module, preserving the same algorithm logic. No architectural refactoring during the conversion.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Parallelism | Single-threaded first, Numba later | Get correctness first; optimize with Numba in a follow-up |
| Intermediate serialization (`.jlse`) | Drop entirely | All in-memory now — no need for Julia serialization |
| API shape | Keep same methods (`.CO()`, `.SynthPop()`, `.Export()`, `.run_all()`) | Minimize API churn; refactor later |
| Class name | `RunJulia` → `RunPython` | Reflects the new implementation |
| Export behavior | Write CSV/MTX to disk AND return in-memory | Downstream `ForStarsim` reads files; in-memory access is a bonus |
| IPF implementation | Use numpy-based IPF, validate against existing `ipfn.py` | Likely more optimized; must verify equivalent results |

## Module Mapping

| Julia file | New Python module | Purpose |
|---|---|---|
| `utils.jl` | `utils.py` | Dataclasses (`PersonData`, `Household`, `GQres`, `Indexer`), helper functions (`lrRound`, `drawCounts`, `ranges`, etc.) |
| `fileutils.jl` | *(absorbed — not needed)* | Julia serialization/CSV wrappers; Python uses `pd.read_csv()` directly |
| `CO.jl` | `co.py` | Simulated annealing combinatorial optimization |
| `households.jl` | `households.py` | Household & person generation, group quarters |
| `schools.jl` | `schools.py` | School assignment |
| `workplaces.jl` | `workplaces.py` | Workplace generation, worker assignment, commute matrix IPF |
| `netw.jl` | `networks.py` | Contact network generation (SBM, small-world, complete graphs) |
| `export_synthpop.jl` + `export_network.jl` | `export.py` | Export population/network to CSV/MTX |
| `synthpop.jl` | *(orchestration moves into `RunPython`)* | Thin orchestrator; logic absorbed by the class |

## Data Structures

### PersonData
```python
@dataclass
class PersonData:
    hh: tuple[int, int]           # (hh_num, cbg_index)
    sample: int
    age: int
    working: bool
    commuter: bool
    com_cat: int | None           # workplace category
    com_inc: int | None           # income category
    sch_grade: str | None
    sch_public: bool | None
    sch_private: bool | None
    female: bool | None
    race_black_alone: bool | None
    white_non_hispanic: bool | None
    hispanic: bool | None
```

### Household
```python
@dataclass
class Household:
    sample: int
    people: list[tuple[int, int, int]]  # list of Pkey = (p_num, hh_num, cbg_key)
```

### GQres
```python
@dataclass
class GQres:
    type: str
    residents: list[tuple[int, int, int]]
```

### Indexer
```python
class Indexer:
    """Auto-assigns sequential integer IDs to new keys."""
    def __init__(self):
        self.i = 0

    def __call__(self, d: dict, k):
        if k in d:
            return d[k]
        self.i += 1
        d[k] = self.i
        return self.i
```

### Keys
Plain tuples as dict keys:
- `Pkey = tuple[int, int, int]` — (person_num, hh_num, cbg_key)
- `Hkey = tuple[int, int]` — (hh_num, cbg_key)
- `GQkey = tuple[int, int]` — (type_idx, cbg_key)
- `WRKkey = tuple[int, int, str]` — (work_num, cat_idx, dest_code)

### Sparse matrices
`scipy.sparse` (CSR/CSC) replaces Julia's `SparseArrays`.

### Graphs
`networkx` replaces Julia's `Graphs.jl`.

## Algorithm Translation

### Simulated Annealing (co.py)
- Direct translation of `anneal()`, single-threaded, using numpy
- `summarize()` → `pop[idxs, :].sum(axis=0)`
- `FTdist()` → `np.sum((np.sqrt(v1 + 1) - np.sqrt(v2 + 1)) ** 2)`
- `mutate!()` / `revert!()` → mutate list in-place, return undo info
- Hierarchical retry logic (puma → county → cbsa → urbanization) stays the same
- `SharedMatrix` / `@distributed` / `pmap` all removed — regular numpy array and for-loop

### IPF for Commute Matrices (workplaces.py)
- Use a numpy-based IPF implementation
- Validate that results match the existing `ipfn.py` before switching
- The IPF here is 2-margin on a 2D matrix — straightforward

### Network Generation (networks.py)
- `networkx.stochastic_block_model()` replaces `Graphs.stochastic_block_model()`
- `networkx.watts_strogatz_graph()` replaces `Graphs.watts_strogatz()`
- `networkx.complete_graph()` replaces `Graphs.complete_graph()`
- Edge extraction and sparse matrix construction via `scipy.sparse`

### Matrix Market Export (export.py)
- `scipy.io.mmwrite()` replaces Julia's `MatrixMarket.mmwrite()`

## API Changes

### RunPython class
Same interface as the old `RunJulia`, minus the Julia dependency:

```python
class RunPython:
    def __init__(self, output_dir=None):
        # No julia_env_path needed
        ...

    def CO(self):
        # Calls co.process_counties(), stores results on self
        ...

    def SynthPop(self):
        # Calls households, schools, workplaces, networks — in-memory pipeline
        ...

    def Export(self):
        # Writes CSV/MTX to pop_export/, also returns data in-memory
        ...

    def run_all(self):
        self.CO()
        self.SynthPop()
        self.Export()
```

### __init__.py
```python
from .config import WriteConfig
from .download_data import DownloadData
from .census import ProcessData
from .run_python import RunPython
from .geopops_starsim import ForStarsim
```

### pyproject.toml
- Add dependencies: `scipy`, `networkx`
- Remove `julia/*` from package-data (after Julia code is retired)
- `julia_env_path` in config becomes optional/ignored

## Output Compatibility

The `pop_export/` directory structure and file formats remain identical:

```
pop_export/
├── cbg_idxs.csv
├── hh.csv
├── people.csv
├── sch_students.csv
├── sch_workers.csv
├── gqs.csv
├── gq_residents.csv
├── gq_workers.csv
├── company_workers.csv
├── outside_workers.csv
├── adj_upper_triang_hh.mtx
├── adj_upper_triang_non_hh.mtx
├── adj_upper_triang_wp.mtx
├── adj_upper_triang_sch.mtx
├── adj_upper_triang_gq.mtx
├── adj_mat_keys.csv
├── adj_dummy_keys.csv
└── adj_out_workers.csv
```

`ForStarsim` reads from these files — no changes needed there.

## Validation Strategy

1. Run Julia on a small test geography and save all outputs
2. Run Python equivalent on the same geography with the same random seed
3. Compare:
   - **CO scores:** same ballpark (not identical — different RNG streams)
   - **Households/schools/workplaces:** given identical CO results, counts and assignments should match
   - **Networks:** degree distributions and edge counts should be statistically similar
   - **Exports:** CSV/MTX files structurally identical (same columns, same row counts)

## Future Work (Out of Scope)

- Numba parallelization of simulated annealing
- API refactoring (collapsing pipeline steps, richer return types)
- Removing Julia source files from the repository
- Performance benchmarking Python vs Julia
