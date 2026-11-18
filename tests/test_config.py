"""Check configuration errors before making provider requests."""

import pytest

from config import get_model


def test_bedrock_requires_model_id(monkeypatch):
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)

    with pytest.raises(ValueError, match="Set BEDROCK_MODEL_ID"):
        get_model()
