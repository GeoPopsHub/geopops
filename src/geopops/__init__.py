"""GeoPops: geographically and demographically realistic synthetic populations.

Typical use::

    import geopops as gp

    cfg = gp.make_config(path="data", geos=["45083"], main_year=2019,
                         commute_states=["45", "37"], use_pums=["45", "37"],
                         random_seed=42)

    gp.download_data(cfg)     # fetch Census/PUMS/LODES/school data (slow, cached)
    gp.process_data(cfg)      # build the CO targets and sample pools
    pop = gp.generate_pop(cfg, seed=42)
    ppl = gp.to_starsim_people(pop.pop_export_dir)

or, all at once::

    pop = gp.run(cfg, seed=42)
"""
from .exceptions import (GeoPopsError, ConfigError, DownloadError, DataError,
                         PipelineStateError)
from .config import make_config, load_config, save_config, validate_config
from .sources import DownloadData, download_data, download
from .census import ProcessData, process_data, quality_check
from .population import GeneratePop, generate_pop
from .pipeline import run
from .starsim_bridge import (GPNetwork, SubgroupTracking, to_starsim_people,
                              starsim_network, starsim_networks, load_network_edges)

__version__ = "0.1.8"

__all__ = [
    "__version__",
    # Errors
    "GeoPopsError", "ConfigError", "DownloadError", "DataError", "PipelineStateError",
    # Config
    "make_config", "load_config", "save_config", "validate_config",
    # Pipeline
    "download_data", "process_data", "generate_pop", "quality_check", "run",
    # Starsim bridge
    "to_starsim_people", "starsim_network", "starsim_networks", "load_network_edges",
    "GPNetwork", "SubgroupTracking",
    # Step objects, for running or inspecting individual stages
    "DownloadData", "ProcessData", "GeneratePop", "download",
]
