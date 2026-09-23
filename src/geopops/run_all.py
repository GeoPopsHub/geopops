"""Top-level pipeline orchestrator for GeoPops."""

from .write_config import write_config, load_config, update_config_values
from .download_data import download_data
from .process_data import process_data
from .generate_pop import generate_pop
from . import for_starsim

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


def _build_effective_config(config_dict=None, pars=None, base_dir=None):
    """Resolve the config for a full run.

    `pars` supplies partial overrides; `config_dict` is treated as a complete
    config and used as-is.
    """
    pars = pars or {}
    if config_dict is not None:
        config = config_dict
    else:
        config = load_config(base_dir)
        update_config_values(
            config,
            census_api_key=pars.get("census_api_key"),
            main_year=pars.get("main_year"),
            geos=pars.get("geos"),
            commute_states=pars.get("commute_states"),
            use_pums=pars.get("use_pums"),
            path=pars.get("path"),
            julia_env_path=pars.get("julia_env_path"),
        )
    # Backfill required table-code keys when config templates are minimal.
    config.setdefault("acs_required", DEFAULT_ACS_REQUIRED.copy())
    config.setdefault("dec_required", DEFAULT_DEC_REQUIRED.copy())
    return config


def run_all(config=None, pars=None, base_dir=None, random_seed=None, verbose=1):
    """Run the full GeoPops workflow with a single call.

    write_config -> download_data -> process_data -> generate the population ->
    build the Starsim people and networks.

    Args:
        config (dict, optional): A complete config dict, used as-is. If not
            provided, loads via load_config(base_dir) and applies `pars`.
        pars (dict, optional): Partial overrides (census_api_key, main_year,
            geos, commute_states, use_pums, path, julia_env_path).
        base_dir (str, optional): Base directory for relative paths.
        random_seed (int, optional): Master seed for population generation.
            Takes precedence over config["random_seed"], matching Population.
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.

    Returns:
        Population: the completed run, holding all pipeline intermediates.

    Example::

        pop = geopops.run_all(pars={"path": "pops/sc_45083"}, random_seed=42)
        pop.people, pop.households, pop.adj_hh
    """
    def _log(msg):
        if verbose:
            print(msg)

    _log("Generating population with run_all()")

    effective_config = _build_effective_config(
        config_dict=config, pars=pars, base_dir=base_dir)

    write_config(config_dict=effective_config, base_dir=base_dir)

    download_data(
        effective_config,
        base_dir=base_dir,
        verbose=verbose,
    )

    process_data(
        effective_config,
        base_dir=base_dir,
        verbose=verbose,
    )

    pop = generate_pop(
        config=effective_config,
        base_dir=base_dir,
        random_seed=random_seed,
        verbose=verbose,
    )

    # Pass the same path to both sides: network() used to read the package
    # config regardless of what people() was given, so the two could read
    # different populations. A loop also means a new network can't be added
    # while forgetting to pass the path.
    pop_path = effective_config.get('path')
    for_starsim.people(config_dict=effective_config, base_dir=base_dir,
                       path=pop_path)
    for _net_name in ('homenet', 'schoolnet', 'worknet', 'gqnet'):
        for_starsim.network(name=_net_name, edge_weight=1.0, path=pop_path)

    _log("")
    _log("Population generation complete")
    return pop
