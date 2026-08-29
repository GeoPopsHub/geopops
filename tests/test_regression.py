"""Golden-output regression test.

Pins the pipeline's output for a fixed seed so refactors can be checked for
behaviour changes. When a change is *intended*, regenerate the baseline::

    python tests/test_regression.py --update

and review the diff to ``tests/golden_hashes.json`` as part of the change.
"""
import hashlib
import json
import os
import sys

import pytest

import geopops
from conftest import requires_fixture_data, FIXTURE_PARS, FIXTURE_DATA

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_hashes.json")
SEED = 42
CO_MAXGENS = 5000

#: Outputs whose content should be fully determined by the seed
TRACKED = [
    "cbg_idxs.csv", "hh.csv", "people.csv", "sch_students.csv", "gqs.csv",
    "gq_residents.csv", "adj_mat_keys.csv", "adj_upper_triang_hh.mtx",
    "sch_workers.csv", "company_workers.csv", "gq_workers.csv",
    "outside_workers.csv", "adj_upper_triang_sch.mtx", "adj_upper_triang_wp.mtx",
    "adj_upper_triang_gq.mtx",
]


def _build_config(run_dir):
    fixture_cfg = os.path.join(FIXTURE_DATA, "config.json")
    template = json.load(open(fixture_cfg)) if os.path.exists(fixture_cfg) else None
    cfg = geopops.make_config(template=template, **FIXTURE_PARS)
    cfg["path"] = run_dir
    cfg["CO_maxgens"] = CO_MAXGENS
    return cfg


def compute_hashes(run_dir):
    """Generate a population and hash its outputs."""
    pop = geopops.generate_pop(_build_config(run_dir), seed=SEED, verbose=0)
    out = {}
    for name in TRACKED:
        path = os.path.join(pop.pop_export_dir, name)
        out[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return out


@pytest.mark.slow
@requires_fixture_data
def test_outputs_match_golden(run_dir):
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(f"No baseline at {GOLDEN_PATH}; create it with "
                    f"`python tests/test_regression.py --update`")
    golden = json.load(open(GOLDEN_PATH))
    if golden.get("_seed") != SEED or golden.get("_co_maxgens") != CO_MAXGENS:
        pytest.skip("Baseline was recorded with different settings; regenerate it.")

    actual = compute_hashes(run_dir)
    changed = [n for n in TRACKED if golden.get(n) != actual.get(n)]
    assert not changed, (
        "Pipeline output changed for these files: " + ", ".join(changed) +
        "\nIf this is intended, regenerate the baseline with "
        "`python tests/test_regression.py --update` and review the diff."
    )


@pytest.mark.slow
@requires_fixture_data
def test_same_seed_gives_the_same_population(tmp_path):
    """Two runs with one seed must agree exactly."""
    import shutil
    hashes = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        shutil.copytree(FIXTURE_DATA, d)
        shutil.rmtree(d / "pop_export", ignore_errors=True)
        hashes.append(compute_hashes(str(d)))
    differing = [n for n in TRACKED if hashes[0][n] != hashes[1][n]]
    assert not differing, f"Not reproducible under a fixed seed: {differing}"


if __name__ == "__main__":
    import shutil
    import tempfile

    if "--update" not in sys.argv:
        print(__doc__)
        sys.exit(1)
    tmp = tempfile.mkdtemp()
    run_dir = os.path.join(tmp, "data")
    shutil.copytree(FIXTURE_DATA, run_dir)
    shutil.rmtree(os.path.join(run_dir, "pop_export"), ignore_errors=True)
    result = {"_seed": SEED, "_co_maxgens": CO_MAXGENS, **compute_hashes(run_dir)}
    with open(GOLDEN_PATH, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Wrote baseline for {len(TRACKED)} files to {GOLDEN_PATH}")
