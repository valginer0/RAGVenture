"""
Tests for the StartupIdeaGenerator class.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from rag_startups.idea_generator.generator import RateLimitError, StartupIdeaGenerator


@pytest.fixture
def mock_inference_client():
    """Fixture to provide a mock InferenceClient"""
    with patch("rag_startups.idea_generator.generator.InferenceClient") as mock_client:
        # Configure the mock client's behavior
        instance = mock_client.return_value
        instance.text_generation.return_value = """
Startup Idea #1:
Name: TestStartup
Problem/Opportunity: Test problem statement
Solution: Test solution description
Target Market: Test market
Unique Value:
• Test value 1
• Test value 2
• Test value 3
"""

        yield instance


@pytest.fixture
def generator(mock_inference_client):
    """Fixture to provide a StartupIdeaGenerator instance with mocked client"""
    with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "test_token"}):
        return StartupIdeaGenerator()


def test_generate_success(generator, mock_inference_client):
    """Test successful idea generation"""
    response, insights = generator.generate(num_ideas=1, include_market_analysis=False)
    assert response is not None
    assert "TestStartup" in response
    assert insights is None  # No market analysis requested
    mock_inference_client.text_generation.assert_called_once()


def test_generate_rate_limit(generator, mock_inference_client):
    """Test rate limit handling"""
    # Fill up the rate limit
    generator.max_requests_per_hour = 1
    generator._update_rate_limit()  # Add a request timestamp directly

    # Next request should raise RateLimitError
    with pytest.raises(RateLimitError):
        generator.generate(num_ideas=1)


def test_generate_invalid_num_ideas(generator):
    """Test validation of num_ideas parameter"""
    with pytest.raises(ValueError):
        generator.generate(num_ideas=0)

    with pytest.raises(ValueError):
        generator.generate(num_ideas=6)


def test_generate_api_error(generator, mock_inference_client):
    """Test handling of API errors"""
    mock_inference_client.text_generation.side_effect = Exception("API Error")

    with pytest.raises(Exception):
        generator.generate(num_ideas=1)


def test_rate_limit_cleanup(generator):
    """Test that old timestamps are cleaned up"""
    # Add some old timestamps
    old_time = datetime.now() - timedelta(hours=2)
    generator.request_timestamps = [old_time]

    # Check rate limit should clean old timestamps
    assert generator._check_rate_limit() is True
    assert len(generator.request_timestamps) == 0
