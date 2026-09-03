"""Unit tests for the download layer. No network access: the fetchers are patched."""
import warnings
from unittest import mock

import pytest

from geopops import sources
from geopops.exceptions import DownloadError


@pytest.fixture(autouse=True)
def _secure_by_default():
    """Every test starts with the insecure fallback off, and leaves it off."""
    sources.set_allow_insecure_downloads(False)
    yield
    sources.set_allow_insecure_downloads(False)


@pytest.fixture
def dst(tmp_path):
    return str(tmp_path / "out.bin")


def _writes(content=b"ok"):
    def fetch(src, dst, headers, mode, verify):
        with open(dst, "wb") as f:
            f.write(content)
    return fetch


class TestEffectiveACSYear:
    """Some ACS tables have no 2023/2024 vintage and fall back to 2022."""

    @pytest.mark.parametrize("code,year,expected", [
        ("B09019", 2023, 2022),
        ("B09020", 2024, 2022),
        ("B01001", 2023, 2023),   # not in the fallback set
        ("B09019", 2019, 2019),   # year has data
    ])
    def test_fallback(self, code, year, expected):
        assert sources._effective_acs_year(code, year) == expected


class TestDownload:
    def test_success_and_verifies_tls_by_default(self, dst):
        with mock.patch.object(sources, "_fetch_requests", side_effect=_writes()) as f:
            assert sources.download("http://example/x", dst) == 0
        assert open(dst, "rb").read() == b"ok"
        assert f.call_args[1]["verify"] is True

    def test_raises_instead_of_exiting(self, dst):
        """Regression: this path used to call exit(1), killing the host process."""
        with mock.patch.object(sources, "_fetch_requests", side_effect=OSError("boom")):
            with pytest.raises(DownloadError, match="after 2 attempts"):
                sources.download("http://example/x", dst, retries=2)

    def test_retries_then_succeeds(self, dst):
        attempts = []

        def flaky(src, d, headers, mode, verify):
            attempts.append(1)
            if len(attempts) < 2:
                raise OSError("transient")
            _writes()(src, d, headers, mode, verify)

        with mock.patch.object(sources, "_fetch_requests", side_effect=flaky):
            assert sources.download("http://example/x", dst, retries=3) == 0
        assert len(attempts) == 2

    def test_unknown_backend(self, dst):
        with pytest.raises(ValueError, match="Unknown download backend"):
            sources.download("http://example/x", dst, backend="carrier-pigeon")

    def test_curl_cffi_backend_is_selectable(self, dst):
        with mock.patch.object(sources, "_fetch_curl_cffi", side_effect=_writes()) as f:
            sources.download("http://example/x", dst, backend="curl_cffi")
        assert f.called

    def test_text_mode_is_passed_through(self, dst):
        with mock.patch.object(sources, "_fetch_requests", side_effect=_writes()) as f:
            sources.download("http://example/x", dst, mode="text")
        assert f.call_args[0][3] == "text"


class TestInsecureFallback:
    def test_off_by_default_with_an_actionable_message(self, dst):
        err = OSError("SSL: certificate verify failed")
        with mock.patch.object(sources, "_fetch_requests", side_effect=err):
            with pytest.raises(DownloadError, match="allow_insecure_downloads"):
                sources.download("http://example/x", dst, retries=1)

    def test_never_retries_unverified_unless_enabled(self, dst):
        calls = []

        def fetch(src, d, headers, mode, verify):
            calls.append(verify)
            raise OSError("SSL: certificate verify failed")

        with mock.patch.object(sources, "_fetch_requests", side_effect=fetch):
            with pytest.raises(DownloadError):
                sources.download("http://example/x", dst, retries=1)
        assert calls == [True], "an unverified retry happened without opt-in"

    def test_engages_and_warns_when_enabled(self, dst):
        calls = []

        def fetch(src, d, headers, mode, verify):
            calls.append(verify)
            if verify:
                raise OSError("SSL: certificate verify failed")
            _writes()(src, d, headers, mode, verify)

        sources.set_allow_insecure_downloads(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(sources, "_fetch_requests", side_effect=fetch):
                assert sources.download("http://example/x", dst, retries=1) == 0
        assert calls == [True, False]
        assert any("TLS verification disabled" in str(w.message) for w in caught)

    def test_non_tls_errors_do_not_trigger_it(self, dst):
        calls = []

        def fetch(src, d, headers, mode, verify):
            calls.append(verify)
            raise OSError("404 Not Found")

        sources.set_allow_insecure_downloads(True)
        with mock.patch.object(sources, "_fetch_requests", side_effect=fetch):
            with pytest.raises(DownloadError):
                sources.download("http://example/x", dst, retries=1)
        assert calls == [True]


class TestNoImportTimeSideEffects:
    def test_warning_filters_are_not_globally_disabled(self):
        """Regression: importing geopops used to call urllib3.disable_warnings()."""
        import urllib3
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("probe", urllib3.exceptions.InsecureRequestWarning, stacklevel=1)
        assert any("probe" in str(w.message) for w in caught)


class TestFipsInfo:
    def test_fips_to_abbreviation(self):
        assert sources.fips_info("45")["abbr"] == "SC"
        assert sources.fips_info(["45", "37"])["abbr"] == ["SC", "NC"]

    def test_reverse(self):
        assert sources.fips_info("SC", reverse=True)["fips"] == "45"

    def test_unknown_code(self):
        assert sources.fips_info("99")["abbr"] is None
