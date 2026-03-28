"""
RunPython orchestrator class — pure-Python replacement for RunJulia.
Calls co, households, schools, workplaces, networks, and export modules.
"""
import os
import json
from . import co, households, schools, workplaces, networks, export

# Parent package directory (src/geopops/) where config.json lives
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    config_path = os.path.join(PACKAGE_DIR, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


class RunPython:
    """Orchestrates the synthetic population pipeline in pure Python.
    Same API as RunJulia: CO(), SynthPop(), Export(), run_all().
    """

    def __init__(self, output_dir=None):
        config = load_config()
        if output_dir is not None:
            self.data_dir = output_dir
        else:
            self.data_dir = config.get("path", PACKAGE_DIR)
        self.config = config
        print(f"Using data directory: {self.data_dir}")

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

    def CO(self):
        """Run combinatorial optimization."""
        print("=" * 60)
        print("Running combinatorial optimization")
        print("=" * 60)
        self.co_results, self.co_scores = co.process_counties(self.data_dir)

    def SynthPop(self):
        """Generate synthetic population (households, schools, workplaces, networks)."""
        if self.co_results is None:
            raise RuntimeError("CO() must be run before SynthPop()")

        print("=" * 60)
        print("Creating people, households, and group quarters")
        print("=" * 60)
        self.cbgs, self.people, self.households, self.gqs, self.gq_summary = \
            households.generate_people(self.co_results, self.data_dir)

        print("=" * 60)
        print("Creating schools")
        print("=" * 60)
        self.sch_students = schools.generate_schools(
            self.people, self.cbgs, self.data_dir)

        print("=" * 60)
        print("Creating workplaces")
        print("=" * 60)
        (self.company_workers, self.sch_workers, self.gq_workers,
         self.outside_workers, self.dummies) = \
            workplaces.generate_jobs_and_workers(
                self.people, self.cbgs, self.gqs,
                self.co_results, self.gq_summary, self.data_dir)

        print("=" * 60)
        print("Creating networks")
        print("=" * 60)
        (self.adj_hh, self.adj_non_hh, self.adj_wp, self.adj_sch, self.adj_gq,
         self.adj_mat_keys, self.adj_dummy_keys, self.adj_out_workers) = \
            networks.generate_networks(
                self.people, self.households, self.gqs, self.sch_students,
                self.company_workers, self.sch_workers, self.gq_workers,
                self.outside_workers, self.dummies, self.config)

        print("done with SynthPop")

    def Export(self):
        """Export population and networks to CSV/MTX files."""
        if self.people is None:
            raise RuntimeError("SynthPop() must be run before Export()")

        print("=" * 60)
        print("Exporting population data")
        print("=" * 60)
        export.export_synthpop(
            self.data_dir, self.cbgs, self.households, self.people,
            self.sch_students, self.sch_workers, self.gqs,
            self.gq_workers, self.company_workers, self.outside_workers)

        print("Exporting network data")
        export.export_networks(
            self.data_dir, self.adj_hh, self.adj_non_hh, self.adj_wp,
            self.adj_sch, self.adj_gq, self.adj_mat_keys,
            self.adj_dummy_keys, self.adj_out_workers)

    def run_all(self):
        """Run the complete pipeline: CO -> SynthPop -> Export."""
        self.CO()
        self.SynthPop()
        self.Export()
