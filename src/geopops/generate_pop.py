"""
GeneratePop orchestrator class — pure-Python replacement for RunJulia.
Calls co, households, schools, workplaces, networks, and export modules.
"""
import os
import json
import numpy as np
from . import co, households, schools, workplaces, networks, export

# Package directory (src/geopops/) where config.json lives
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(base_dir=None):
    cfg_dir = base_dir if base_dir is not None else PACKAGE_DIR
    config_path = os.path.join(cfg_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class GeneratePop:
    """Orchestrates the synthetic population pipeline in pure Python.
    Same API as RunJulia: CO(), SynthPop(), Export(), run_all().
    """

    def __init__(self, config_dict=None, base_dir=None, output_dir=None,
                 random_seed=None, verbose=1, auto_run=False, run_all=None):
        self.verbose = verbose
        self.base_dir = base_dir if base_dir is not None else PACKAGE_DIR
        config = config_dict if config_dict is not None else load_config(self.base_dir)
        if output_dir is not None:
            self.data_dir = output_dir
        else:
            self.data_dir = config.get("path", self.base_dir)
        self.config = config
        self.random_seed = random_seed if random_seed is not None else config.get("random_seed")
        self._stage_random_seeds = self._make_stage_random_seeds(self.random_seed)
        self._log(f"Using data directory: {self.data_dir}")

        # Pipeline state (populated by each stage)
        self.co_results = None
        self.co_scores = None
        self.cbgs = None
        self.people = None
        self.households = None
        self.gqs = None
        self.gq_summary = None
        self.sch_students = None
        self.company_workers = None
        self.sch_workers = None
        self.gq_workers = None
        self.outside_workers = None
        self.dummies = None
        self.adj_hh = None
        self.adj_non_hh = None
        self.adj_wp = None
        self.adj_sch = None
        self.adj_gq = None
        self.adj_mat_keys = None
        self.adj_dummy_keys = None
        self.adj_out_workers = None

        if run_all is not None:
            auto_run = run_all
        if auto_run:
            self.run_all()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _make_stage_random_seeds(random_seed):
        """Derive deterministic per-stage seeds from one master random seed."""
        if random_seed is None:
            return {
                "co": None,
                "households": None,
                "schools": None,
                "workplaces": None,
                "networks": None,
            }
        child_states = np.random.SeedSequence(random_seed).spawn(5)
        labels = ["co", "households", "schools", "workplaces", "networks"]
        return {
            k: int(child_states[i].generate_state(1, dtype=np.uint32)[0])
            for i, k in enumerate(labels)
        }

    def CO(self):
        """Run combinatorial optimization."""
        self._log("=" * 60)
        self._log("Running combinatorial optimization")
        self._log("=" * 60)
        self.co_results, self.co_scores = co.process_counties(
            self.data_dir, random_seed=self._stage_random_seeds["co"])

    def SynthPop(self):
        """Generate synthetic population (households, schools, workplaces, networks)."""
        if self.co_results is None:
            raise RuntimeError("CO() must be run before SynthPop()")

        self._log("=" * 60)
        self._log("Creating people, households, and group quarters")
        self._log("=" * 60)
        self.cbgs, self.people, self.households, self.gqs, self.gq_summary = \
            households.generate_people(
                self.co_results, self.data_dir,
                random_seed=self._stage_random_seeds["households"])

        self._log("=" * 60)
        self._log("Creating schools")
        self._log("=" * 60)
        self.sch_students = schools.generate_schools(
            self.people, self.cbgs, self.data_dir,
            random_seed=self._stage_random_seeds["schools"])

        self._log("=" * 60)
        self._log("Creating workplaces")
        self._log("=" * 60)
        (self.company_workers, self.sch_workers, self.gq_workers,
         self.outside_workers, self.dummies) = \
            workplaces.generate_jobs_and_workers(
                self.people, self.cbgs, self.gqs,
                self.co_results, self.gq_summary, self.data_dir,
                random_seed=self._stage_random_seeds["workplaces"])

        self._log("=" * 60)
        self._log("Creating networks")
        self._log("=" * 60)
        (self.adj_hh, self.adj_non_hh, self.adj_wp, self.adj_sch, self.adj_gq,
         self.adj_mat_keys, self.adj_dummy_keys, self.adj_out_workers) = \
            networks.generate_networks(
                self.people, self.households, self.gqs, self.sch_students,
                self.company_workers, self.sch_workers, self.gq_workers,
                self.outside_workers, self.dummies, self.config,
                random_seed=self._stage_random_seeds["networks"])

        self._log("done with SynthPop")

    def Export(self):
        """Export population and networks to CSV/MTX files."""
        if self.people is None:
            raise RuntimeError("SynthPop() must be run before Export()")

        self._log("=" * 60)
        self._log("Exporting population data")
        self._log("=" * 60)
        export.export_synthpop(
            self.data_dir, self.cbgs, self.households, self.people,
            self.sch_students, self.sch_workers, self.gqs,
            self.gq_workers, self.company_workers, self.outside_workers)

        self._log("Exporting network data")
        export.export_networks(
            self.data_dir, self.adj_hh, self.adj_non_hh, self.adj_wp,
            self.adj_sch, self.adj_gq, self.adj_mat_keys,
            self.adj_dummy_keys, self.adj_out_workers)

    def run_all(self):
        """Run the complete pipeline: CO -> SynthPop -> Export."""
        self.CO()
        self.SynthPop()
        self.Export()

