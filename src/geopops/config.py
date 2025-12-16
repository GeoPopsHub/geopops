import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    cfg_path = os.path.join(BASE_DIR, "config.json")
    # if not os.path.exists(cfg_path):
    #     raise FileNotFoundError(f"config.json file not found at {cfg_path}. Please create this file with the required configuration.")
    with open(cfg_path, "r") as f:
        return json.load(f)


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
                 config_dict=None,
                 base_dir=None):
        self.base_dir = base_dir if base_dir is not None else BASE_DIR
        # Load base template from the package directory unless a dict is provided
        self.template_config_path = os.path.join(self.base_dir, "config.json")
        if path is not None:
            # If path is a directory, append 'config.json' to it
            if os.path.isdir(path):
                self.path = os.path.join(path, "config.json")
            else:
                self.path = path
        else:
            self.path = self.template_config_path
        self.config = config_dict if config_dict is not None else load_config()
        self.overrides = {
            "census_api_key": census_api_key,
            "main_year": main_year,
            "geos": geos,
            "commute_states": commute_states,
            "use_pums": use_pums,
            "path": path,
            "julia_env_path": julia_env_path,
        }
        
        self.run_all()

    def run_all(self):
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
        print("Updated config.json with new parameters")

    def get_pars(self):
        with open(self.path, "r") as f:
            cfg = json.load(f)
        print(json.dumps(cfg, indent=2))


def main():
    runner = WriteConfig()
    runner.run_all()


if __name__ == "__main__":
    main()