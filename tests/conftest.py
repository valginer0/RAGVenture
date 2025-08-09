from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_hf_model_info_autouse():
    """Mock huggingface_hub.model_info used by CLI preflight to avoid network/auth.
    Applies to all tests importing rag_startups.cli.
    """
    with patch("rag_startups.cli.model_info") as mi:
        mi.return_value = object()
        yield mi
