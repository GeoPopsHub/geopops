"""Unit tests for config loading, merging, overriding, and validation."""
import json

import pytest

import geopops
from geopops import config as cfgmod
from geopops.exceptions import ConfigError


class TestMakeConfig:
    def test_applies_overrides(self):
        c = geopops.make_config(path="out", geos=["45083"], main_year=2019, random_seed=1)
        assert c["path"] == "out" and c["geos"] == ["45083"] and c["main_year"] == 2019

    def test_derives_decennial_year(self):
        assert geopops.make_config(path="o", geos=["1"], main_year=2019, random_seed=1)["decennial_year"] == 2010
        assert geopops.make_config(path="o", geos=["1"], main_year=2021, random_seed=1)["decennial_year"] == 2020

    def test_backfills_required_tables(self):
        c = geopops.make_config(template={"path": "o", "geos": ["1"], "main_year": 2019,
                                          "random_seed": 1})
        assert "B01001" in c["acs_required"]
        assert c["dec_required"] == ["P43", "P18"]

    def test_rejects_unknown_override(self):
        # A typo used to be silently ignored, producing a run with default settings
        with pytest.raises(ConfigError, match="geoss"):
            geopops.make_config(path="o", geos=["1"], main_year=2019, geoss=["45"])

    def test_does_not_mutate_the_template(self):
        template = {"path": "o", "geos": ["1"], "main_year": 2019, "random_seed": 1}
        geopops.make_config(template=template, main_year=2021)
        assert template["main_year"] == 2019

    def test_warns_when_unseeded(self):
        with pytest.warns(UserWarning, match="reproducible"):
            geopops.make_config(path="o", geos=["1"], main_year=2019)


class TestValidation:
    @pytest.mark.parametrize("missing", ["path", "geos", "main_year"])
    def test_missing_required_key(self, missing):
        c = {"path": "o", "geos": ["1"], "main_year": 2019, "random_seed": 1}
        del c[missing]
        with pytest.raises(ConfigError, match=missing):
            cfgmod.validate_config(c)

    def test_unparseable_year_is_rejected(self):
        # This used to fall back to 2010 and silently produce wrong-vintage data
        with pytest.raises(ConfigError, match="main_year"):
            cfgmod.compute_decennial_year("twenty-nineteen")

    def test_empty_geos_rejected(self):
        with pytest.raises(ConfigError, match="geos"):
            cfgmod.validate_config({"path": "o", "geos": [], "main_year": 2019,
                                    "random_seed": 1})


class TestSaveLoad:
    def test_round_trip(self, tmp_path, base_config):
        path = cfgmod.save_config(base_config, str(tmp_path / "config.json"))
        assert json.load(open(path))["geos"] == base_config["geos"]

    def test_defaults_to_run_directory_not_the_package(self, tmp_path, base_config):
        base_config["path"] = str(tmp_path / "run")
        path = cfgmod.save_config(base_config)
        assert path == str(tmp_path / "run" / "config.json")
        # crucially, nothing was written into the installed package
        assert cfgmod.BASE_DIR not in path

    def test_directory_target_appends_filename(self, tmp_path, base_config):
        path = cfgmod.save_config(base_config, str(tmp_path))
        assert path.endswith("config.json")

    def test_sanitize_strips_secrets(self, tmp_path, base_config):
        base_config["census_api_key"] = "secret-key-value"
        path = cfgmod.save_config(base_config, str(tmp_path / "t.json"), sanitize=True)
        written = json.load(open(path))
        assert written["census_api_key"] is None
        assert "secret-key-value" not in open(path).read()

    def test_local_overrides_are_merged(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(
            {"path": "o", "geos": ["1"], "main_year": 2019, "CO_maxgens": 10}))
        (tmp_path / "config.local.json").write_text(json.dumps({"CO_maxgens": 999}))
        assert cfgmod.load_config(str(tmp_path))["CO_maxgens"] == 999

    def test_missing_config_raises_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            cfgmod.load_config(str(tmp_path / "nowhere"))


class TestMakeConfigSavesToTheRunDirectory:
    def test_save_writes_next_to_the_output(self, tmp_path):
        c = geopops.make_config(path=str(tmp_path), geos=["45083"], main_year=2019,
                                random_seed=1, save=True)
        assert (tmp_path / "config.json").exists()
        assert json.load(open(tmp_path / "config.json"))["geos"] == c["geos"]

    def test_never_writes_into_the_installed_package(self, tmp_path):
        """Regression: config used to be written into site-packages."""
        before = open(cfgmod.TEMPLATE_PATH).read()
        geopops.make_config(path=str(tmp_path), geos=["45083"], main_year=2019,
                            random_seed=1, save=True)
        assert open(cfgmod.TEMPLATE_PATH).read() == before
