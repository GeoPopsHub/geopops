"""Top-level pipeline orchestrator for GeoPops."""

from .config import make_config, validate_config
from .sources import download_data
from .census import process_data
from .population import generate_pop
from .starsim_bridge import to_starsim_people, starsim_networks


def run(config=None, *, seed=None, download=True, process=True, starsim=True,
        verbose=1, **overrides):
    """Run the whole GeoPops workflow and return the resulting population.

    Args:
        config: a config dict (from :func:`geopops.make_config`). If omitted, one is
            built from the packaged template plus `overrides`.
        seed: master random seed. Overrides ``config['random_seed']``.
        download: fetch raw Census/PUMS/LODES/school data. Set False to reuse data
            already present in the run directory.
        process: rebuild the processed CO targets and sample pools.
        starsim: also build the Starsim ``People`` and network objects.
        verbose: 0 for quiet, 1 for progress logging.
        **overrides: config overrides such as ``geos=``, ``main_year=``, ``path=``.

    Returns:
        GeneratePop: the completed run, with ``people``, ``households``, networks,
        and (if `starsim`) ``ppl`` and ``networks`` attached.

    Example::

        pop = geopops.run(geos=["45083"], main_year=2019, path="data", seed=42)
    """
    if config is None:
        config = make_config(**overrides)
    elif overrides:
        config = make_config(template=config, **overrides)
    else:
        validate_config(config)

    if seed is not None:
        config["random_seed"] = seed

    if verbose:
        print("Generating population with geopops.run()")

    if download:
        download_data(config, verbose=verbose)
    if process:
        process_data(config, verbose=verbose)

    pop = generate_pop(config, seed=config.get("random_seed"), verbose=verbose)

    if starsim:
        pop.ppl = to_starsim_people(pop.pop_export_dir, verbose=verbose)
        pop.networks = starsim_networks(pop.pop_export_dir,
                                        seed=config.get("random_seed") or 0,
                                        verbose=verbose)

    if verbose:
        print("\nPopulation generation complete")
    return pop
