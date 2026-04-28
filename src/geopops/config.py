import json
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _merge_dict(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def load_config(base_dir=None):
    cfg_dir = base_dir if base_dir is not None else BASE_DIR
    cfg_path = os.path.join(cfg_dir, "config.json")
    with open(cfg_path, "r") as f:
        config = json.load(f)

    # Optional untracked local overrides for machine-specific values.
    local_cfg_path = os.path.join(cfg_dir, "config.local.json")
    if os.path.exists(local_cfg_path):
        with open(local_cfg_path, "r") as f:
            local_cfg = json.load(f)
        config = _merge_dict(config, local_cfg)

    return config


def save_config(config, config_path=None):
    cfg_path = config_path if config_path is not None else os.path.join(BASE_DIR, "config.json")
    # Ensure target directory exists when writing to a custom location
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)


def compute_decennial_year(main_year):
    try:
        return 2020 if int(main_year) >= 2020 else 2010
    except Exception:
        return 2010


def update_config_values(config,
                         census_api_key=None,
                         main_year=None,
                         geos=None,
                         commute_states=None,
                         use_pums=None,
                         path=None,
                         julia_env_path=None):
    # Fall back to environment variables for sensitive/user-specific values
    if census_api_key is None:
        census_api_key = os.environ.get("CENSUS_API_KEY")
    if julia_env_path is None:
        julia_env_path = os.environ.get("JULIA_ENV_PATH")

    if census_api_key is not None:
        config["census_api_key"] = census_api_key
    if main_year is not None:
        config["main_year"] = main_year
        config["decennial_year"] = compute_decennial_year(main_year)
    if geos is not None:
        config["geos"] = geos
    if commute_states is not None:
        config["commute_states"] = commute_states
    if use_pums is not None:
        config["use_pums"] = use_pums
    if path is not None:
        config["path"] = path
    if julia_env_path is not None:
        config["julia_env_path"] = julia_env_path
    return config


class WriteConfig:
    def __init__(self,
                 census_api_key=None,
                 main_year=None,
                 geos=None,
                 commute_states=None,
                 use_pums=None,
                 path=None,
                 julia_env_path=None,
                 pars=None,
                 config_dict=None,
                 base_dir=None):
        pars = pars or {}
        self.base_dir = base_dir if base_dir is not None else BASE_DIR
        # Load base template from the package directory unless a dict is provided
        self.template_config_path = os.path.join(self.base_dir, "config.json")
        path = pars.get("path") if path is None else path
        if path is not None:
            # If path is a directory, append 'config.json' to it
            if os.path.isdir(path):
                self.path = os.path.join(path, "config.json")
            else:
                self.path = path
        else:
            self.path = self.template_config_path
        if config_dict is None:
            config_dict = pars.get("config_dict")
        self.config = config_dict if config_dict is not None else load_config(self.base_dir)
        self.overrides = {
            "census_api_key": pars.get("census_api_key") if census_api_key is None else census_api_key,
            "main_year": pars.get("main_year") if main_year is None else main_year,
            "geos": pars.get("geos") if geos is None else geos,
            "commute_states": pars.get("commute_states") if commute_states is None else commute_states,
            "use_pums": pars.get("use_pums") if use_pums is None else use_pums,
            "path": path,
            "julia_env_path": pars.get("julia_env_path") if julia_env_path is None else julia_env_path,
        }
        
        self.run_all()

    def run_all(self):
        print("")
        print("============================================================")
        print("Running WriteConfig()")
        print("============================================================")
        update_config_values(
            self.config,
            census_api_key=self.overrides["census_api_key"],
            main_year=self.overrides["main_year"],
            geos=self.overrides["geos"],
            commute_states=self.overrides["commute_states"],
            use_pums=self.overrides["use_pums"],
            path=self.overrides["path"],
            julia_env_path=self.overrides["julia_env_path"],
        )
        # Save to user-specified path
        save_config(self.config, self.path)
        # Also save to package directory
        save_config(self.config, self.template_config_path)
        print("-- Updated config.json with parameter dictionary")

    def get_pars(self):
        with open(self.path, "r") as f:
            cfg = json.load(f)
        print(json.dumps(cfg, indent=2))


def main():
    runner = WriteConfig()
    runner.run_all()


if __name__ == "__main__":
    main()