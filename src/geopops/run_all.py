"""Top-level pipeline orchestrator for GeoPops."""

from .config import WriteConfig, load_config, update_config_values
from .download_data import DownloadData
from .process_data import ProcessData
from .generate_pop import GeneratePop
from .geopops_starsim import ForStarsim

DEFAULT_ACS_REQUIRED = [
    "B01001",
    "B09019",
    "B09020",
    "C24030",
    "B23025",
    "C24010",
    "B11016",
    "B11012",
    "B23009",
    "B11004",
    "B19001",
    "B22010",
    "B09021",
    "B09018",
    "B11001H",
    "B11001I",
    "B25006",
]
DEFAULT_DEC_REQUIRED = ["P43", "P18"]


class RunAll:
    """Run the full GeoPops workflow with a single call."""

    def __init__(self, config_dict=None, pars=None, base_dir=None, verbose=1, auto_run=True):
        # `pars` provides partial overrides; `config_dict` is treated as a full config.
        self.pars = pars or {}
        self.config_dict = config_dict
        self.base_dir = base_dir
        self.verbose = verbose

        if auto_run:
            self.run_all()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _build_effective_config(self):
        if self.config_dict is not None:
            config = self.config_dict
        else:
            config = load_config(self.base_dir)
            update_config_values(
                config,
                census_api_key=self.pars.get("census_api_key"),
                main_year=self.pars.get("main_year"),
                geos=self.pars.get("geos"),
                commute_states=self.pars.get("commute_states"),
                use_pums=self.pars.get("use_pums"),
                path=self.pars.get("path"),
                julia_env_path=self.pars.get("julia_env_path"),
            )
        # Backfill required table-code keys when config templates are minimal.
        config.setdefault("acs_required", DEFAULT_ACS_REQUIRED.copy())
        config.setdefault("dec_required", DEFAULT_DEC_REQUIRED.copy())
        return config

    def run_all(self):
        self._log("Generating population with RunAll()")

        effective_config = self._build_effective_config()

        WriteConfig(config_dict=effective_config, base_dir=self.base_dir)

        DownloadData(
            config=effective_config,
            base_dir=self.base_dir,
            verbose=self.verbose,
            auto_run=True,
        )

        ProcessData(
            config_dict=effective_config,
            base_dir=self.base_dir,
            verbose=self.verbose,
            auto_run=True,
        )

        GeneratePop(
            config_dict=effective_config,
            base_dir=self.base_dir,
            verbose=self.verbose,
            auto_run=True,
        )

        ForStarsim.People(config_dict=effective_config, base_dir=self.base_dir)
        ForStarsim.GPNetwork(name='homenet', edge_weight=1.0)
        ForStarsim.GPNetwork(name='schoolnet', edge_weight=1.0)
        ForStarsim.GPNetwork(name='worknet', edge_weight=1.0)
        ForStarsim.GPNetwork(name='gqnet', edge_weight=1.0)

        self._log("")
        self._log("Population generation complete")
