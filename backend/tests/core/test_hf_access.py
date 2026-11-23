import sys
import types

import pytest

from app.core.prepare import PrepareError


def _fake_hf(monkeypatch, exc):
    hf = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")

    class GatedRepoError(Exception): ...

    class HfHubHTTPError(Exception):
        def __init__(self, msg, status=None):
            super().__init__(msg)
            self.response = types.SimpleNamespace(status_code=status)

    errors.GatedRepoError = GatedRepoError
    errors.HfHubHTTPError = HfHubHTTPError

    def auth_check(repo):
        if exc is not None:
            raise exc(GatedRepoError, HfHubHTTPError)

    hf.auth_check = auth_check
    hf.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


def test_gated_raises_prepare_error(monkeypatch):
    _fake_hf(monkeypatch, lambda G, H: G("gated"))
    from app.core.backend_kit.hf_access import check_hf_access
    with pytest.raises(PrepareError) as e:
        check_hf_access("org/model")
    assert str(e.value) == "no access to org/model: accept its terms on Hugging Face and set HF_TOKEN"


def test_401_raises_prepare_error(monkeypatch):
    _fake_hf(monkeypatch, lambda G, H: H("unauth", 401))
    from app.core.backend_kit.hf_access import check_hf_access
    with pytest.raises(PrepareError):
        check_hf_access("org/model")


def test_ok_passes(monkeypatch):
    _fake_hf(monkeypatch, None)
    from app.core.backend_kit.hf_access import check_hf_access
    check_hf_access("org/model")
