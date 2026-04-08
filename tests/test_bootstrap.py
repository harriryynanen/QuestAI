from dataclasses import replace

import pytest

from app.bootstrap import _build_llm_client
from app.config import build_app_config
from app.llm.openai_client import OpenAIAppClient


def test_build_llm_client_returns_openai_client_by_default():
    config = build_app_config()

    client = _build_llm_client(config)

    assert isinstance(client, OpenAIAppClient)


def test_build_llm_client_rejects_unsupported_provider():
    config = replace(build_app_config(), llm_provider="stub")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        _build_llm_client(config)
