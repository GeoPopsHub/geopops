"""Top-level pipeline orchestrator for GeoPops."""

from .config import WriteConfig
from .download_data import DownloadData
from .process_data import ProcessData
from .generate_pop import GeneratePop


class RunAll:
    """Run the full GeoPops workflow with a single call."""

    def __init__(self, config_dict=None, base_dir=None, verbose=1, auto_run=True):
        self.config_dict = config_dict
        self.base_dir = base_dir
        self.verbose = verbose

        if auto_run:
            self.run_all()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run_all(self):
        self._log("Writing config")
        WriteConfig(config_dict=self.config_dict, base_dir=self.base_dir)

        self._log("Downloading data")
        DownloadData(
            config=self.config_dict,
            base_dir=self.base_dir,
            verbose=self.verbose,
            auto_run=True,
        )

        self._log("Processing data")
        ProcessData(
            config_dict=self.config_dict,
            base_dir=self.base_dir,
            auto_run=True,
        )

        self._log("Generating synthetic population")
        GeneratePop(
            config_dict=self.config_dict,
            base_dir=self.base_dir,
            verbose=self.verbose,
            auto_run=True,
        )

