"""
GeneratePop orchestrator class — pure-Python replacement for RunJulia.
Calls co, households, schools, workplaces, networks, and export modules.
"""
import os
import json
import numpy as np
from collections import defaultdict
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
        self._log("*** Running GeneratePop.CO() ***")
        self.co_results, self.co_scores = co.process_counties(
            self.data_dir, random_seed=self._stage_random_seeds["co"])

    def _county_from_cbg_idx(self, cbg_idx):
        cbg_code = self._cbg_by_idx.get(cbg_idx)
        return cbg_code[:5] if cbg_code else None

    def _log_people_households_summary(self):
        people_by_county = defaultdict(int)
        hh_by_county = defaultdict(int)
        for person_key in self.people:
            county = self._county_from_cbg_idx(person_key[2])
            if county is not None:
                people_by_county[county] += 1
        for hh_key in self.households:
            county = self._county_from_cbg_idx(hh_key[1])
            if county is not None:
                hh_by_county[county] += 1
        for county in sorted(set(people_by_county) | set(hh_by_county)):
            self._log(f"-- County {county}: {people_by_county[county]} people, {hh_by_county[county]} households")
        self._log(f"-- Total: {len(self.people)} people, {len(self.households)} households")

    def _log_group_quarters_summary(self):
        gq_people_by_county = defaultdict(int)
        gq_by_county = defaultdict(int)
        for person_key in self.people:
            if person_key[1] != 0:
                continue
            county = self._county_from_cbg_idx(person_key[2])
            if county is not None:
                gq_people_by_county[county] += 1
        for gq_key in self.gqs:
            county = self._county_from_cbg_idx(gq_key[1])
            if county is not None:
                gq_by_county[county] += 1
        for county in sorted(set(gq_people_by_county) | set(gq_by_county)):
            self._log(
                f"-- County {county}: {gq_people_by_county[county]} group quarters residents, {gq_by_county[county]} group quarters"
            )
        self._log(
            f"-- Total: {sum(gq_people_by_county.values())} group quarters residents, {len(self.gqs)} group quarters"
        )

    def _log_school_summary(self):
        students_by_county = defaultdict(int)
        schools_by_county = defaultdict(set)
        assigned_schools = set()
        for school_id, student_keys in self.sch_students.items():
            if student_keys:
                assigned_schools.add(school_id)
            for person_key in student_keys:
                county = self._county_from_cbg_idx(person_key[2])
                if county is None:
                    continue
                students_by_county[county] += 1
                schools_by_county[county].add(school_id)
        for county in sorted(set(students_by_county) | set(schools_by_county)):
            self._log(
                f"-- County {county}: {students_by_county[county]} students, {len(schools_by_county[county])} schools"
            )
        self._log(
            f"-- Total: {sum(students_by_county.values())} students, {len(assigned_schools)} schools"
        )
        self._log(f"-- {len(self.sch_students) - len(assigned_schools)} schools assigned 0 students")

    def _log_workplace_summary(self):
        company_count = len(self.company_workers)
        company_workers_n = sum(len(v) for v in self.company_workers.values())
        school_count = len(self.sch_workers)
        school_workers_n = sum(len(v) for v in self.sch_workers.values())
        gq_count = len(self.gq_workers)
        gq_workers_n = sum(len(v) for v in self.gq_workers.values())
        outside_workers_n = sum(len(v) for v in self.outside_workers.values())
        total_workers_n = company_workers_n + school_workers_n + gq_workers_n + outside_workers_n
        total_workplaces_n = company_count + school_count + gq_count
        self._log(f"-- {company_workers_n} company_workers, {company_count} companies")
        self._log(f"-- {school_workers_n} sch_workers, {school_count} schools")
        self._log(f"-- {gq_workers_n} gq_workers, {gq_count} GQs")
        self._log("-- "
                  f"{outside_workers_n} outside_workers (agents who live in geo area but work outside geo area)")
        self._log(f"-- {total_workers_n} total workers, {total_workplaces_n} total workplaces")
        self._log("-- "
                  f"{len(self.dummies)} dummy agents (agents who live and work outside geo area)")

    def _log_network_summary(self):
        work_assoc = self.config.get("income_associativity_coefficient", 0.9)
        work_k = self.config.get("workplace_K", 8)
        sch_assoc = self.config.get("school_associativity_coefficient", 0.9)
        sch_k = self.config.get("school_K", 12)
        gq_k = self.config.get("gq_K", 12)
        rewiring = self.config.get("netw_B", 0.25)
        self._log(f"-- Workplace network (SBM, assortativity={work_assoc}, mean degree={work_k})")
        self._log(f"-- School network (SBM, assortativity={sch_assoc}, mean degree={sch_k})")
        self._log(f"-- GQ network (Small-world, mean degree={gq_k}, rewiring probability={rewiring})")
        self._log("-- Household network (Each household is a complete graph)")

    def SynthPop(self):
        """Generate synthetic population (households, schools, workplaces, networks)."""
        if self.co_results is None:
            raise RuntimeError("CO() must be run before SynthPop()")

        self._log("\n*** Running GeneratePop.SynthPop() ***")
        self.cbgs, self.people, self.households, self.gqs, self.gq_summary = \
            households.generate_people(
                self.co_results, self.data_dir,
                random_seed=self._stage_random_seeds["households"])
        self._cbg_by_idx = {idx: cbg for cbg, idx in self.cbgs.items()}
        self._log("\nGenerating people")
        self._log_people_households_summary()
        self._log("\nGenerating group quarters")
        self._log_group_quarters_summary()

        self.sch_students = schools.generate_schools(
            self.people, self.cbgs, self.data_dir,
            random_seed=self._stage_random_seeds["schools"])
        self._log("\nGenerating schools")
        self._log_school_summary()

        self._log("\nGenerating workplaces")
        self._log("-- Generating OD matrices, exporting interim files")
        self._log("-- processed/od_rows_origins.csv")
        self._log("-- processed/od_columns_dests.csv")
        wp_codes = json.load(open(os.path.join(self.data_dir, "processed", "codes.json"), "r"))
        for ind_code in wp_codes.get("ind_codes", []):
            self._log(f"-- processed/od_{ind_code}.csv.gz")
        (self.company_workers, self.sch_workers, self.gq_workers,
         self.outside_workers, self.dummies) = \
            workplaces.generate_jobs_and_workers(
                self.people, self.cbgs, self.gqs,
                self.co_results, self.gq_summary, self.data_dir,
                random_seed=self._stage_random_seeds["workplaces"])
        self._log_workplace_summary()

        self._log("\nGenerating networks")
        self._log_network_summary()
        (self.adj_hh, self.adj_non_hh, self.adj_wp, self.adj_sch, self.adj_gq,
         self.adj_mat_keys, self.adj_dummy_keys, self.adj_out_workers) = \
            networks.generate_networks(
                self.people, self.households, self.gqs, self.sch_students,
                self.company_workers, self.sch_workers, self.gq_workers,
                self.outside_workers, self.dummies, self.config,
                random_seed=self._stage_random_seeds["networks"])

    def Export(self):
        """Export population and networks to CSV/MTX files."""
        if self.people is None:
            raise RuntimeError("SynthPop() must be run before Export()")

        self._log("\n*** Running GeneratePop.Export() ***")
        self._log("")
        export.export_synthpop(
            self.data_dir, self.cbgs, self.households, self.people,
            self.sch_students, self.sch_workers, self.gqs,
            self.gq_workers, self.company_workers, self.outside_workers,
            verbose=self.verbose)
        export.export_networks(
            self.data_dir, self.adj_hh, self.adj_non_hh, self.adj_wp,
            self.adj_sch, self.adj_gq, self.adj_mat_keys,
            self.adj_dummy_keys, self.adj_out_workers,
            verbose=self.verbose)

    def run_all(self):
        """Run the complete pipeline: CO -> SynthPop -> Export."""
        print("")
        self._log("============================================================")
        self._log("Running GeneratePop()")
        self._log("============================================================")
        self.CO()
        self.SynthPop()
        self.Export()

