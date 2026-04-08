from dataclasses import replace
import builtins
import importlib
import sys

import pytest

from app.bootstrap import _build_llm_client
from app.config import build_app_config


def test_build_llm_client_returns_openai_client_by_default():
    config = build_app_config()

    client = _build_llm_client(config)

    assert client.__class__.__name__ == "OpenAIAppClient"


def test_build_llm_client_rejects_unsupported_provider():
    config = replace(build_app_config(), llm_provider="stub")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        _build_llm_client(config)


def test_build_llm_client_fails_clearly_when_openai_dependency_is_missing(monkeypatch):
    config = build_app_config()

    def fake_import_module(name: str):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        import importlib

        return importlib.import_module(name)

    monkeypatch.setattr("app.llm.openai_client.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="package is not installed"):
        _build_llm_client(config)


def test_importing_api_and_bootstrap_does_not_require_openai(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai" or name.startswith("openai."):
            raise ModuleNotFoundError("No module named 'openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module_name in ("app.api", "app.bootstrap", "app.llm.openai_client"):
        sys.modules.pop(module_name, None)

    bootstrap_module = importlib.import_module("app.bootstrap")
    api_module = importlib.import_module("app.api")

    assert hasattr(bootstrap_module, "_build_llm_client")
    assert hasattr(api_module, "create_api_app")
