"""Tests for market analysis functionality."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from rag_startups.analysis.external_data import (
    BLSData,
    WorldBankData,
    get_industry_analysis,
)
from rag_startups.analysis.market_size import (
    MarketSegment,
    MarketSizeEstimator,
    MarketStage,
)

# Sample test data
SAMPLE_STARTUP_DATA = [
    {"name": "TechCo", "description": "B2B SaaS platform", "valuation": 1000000},
    {"name": "ConsumerApp", "description": "B2C mobile app", "valuation": 2000000},
]


@pytest.fixture
def market_estimator():
    """Create a MarketSizeEstimator instance for testing."""
    return MarketSizeEstimator(SAMPLE_STARTUP_DATA)


@pytest.fixture
def mock_world_bank():
    """Mock World Bank API responses."""
    with patch("wbdata.get_data") as mock:
        mock.return_value = [
            {
                "indicator": {"id": "NY.GDP.MKTP.CD"},
                "value": 20000000000000,  # $20T
            },
            {
                "indicator": {"id": "NV.IND.TOTL.ZS"},
                "value": 25,  # 25% industry
            },
            {
                "indicator": {"id": "NY.GDP.MKTP.KD.ZG"},
                "value": 2.5,  # 2.5% growth
            },
        ]
        yield mock


@pytest.fixture
def mock_bls():
    """Mock BLS API responses."""
    with patch("requests.post") as mock:
        mock.return_value.json.return_value = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {"data": [{"year": "2023", "period": "M12", "value": "1000000"}]}
                ]
            },
        }
        yield mock


def test_market_segment_detection(market_estimator):
    """Test market segment detection."""
    assert market_estimator._determine_segment("B2B SaaS platform") == MarketSegment.B2B
    assert (
        market_estimator._determine_segment("consumer mobile app") == MarketSegment.B2C
    )
    assert (
        market_estimator._determine_segment("enterprise solution")
        == MarketSegment.ENTERPRISE
    )


def test_market_size_estimation(market_estimator, mock_world_bank, mock_bls):
    """Test market size estimation."""
    result = market_estimator.estimate_market_size(
        "B2B SaaS platform", SAMPLE_STARTUP_DATA
    )

    assert result.total_addressable_market > 0
    assert result.serviceable_addressable_market > 0
    assert result.serviceable_obtainable_market > 0
    assert result.confidence_score >= 0 and result.confidence_score <= 1


def test_world_bank_integration(mock_world_bank):
    """Test World Bank data integration."""
    wb = WorldBankData()
    metrics = wb.get_industry_metrics()

    assert "gdp" in metrics
    assert "industry_percentage" in metrics
    assert "growth_rate" in metrics


def test_bls_integration(mock_bls):
    """Test BLS data integration."""
    bls = BLSData()
    data = bls.get_employment_data("5112")

    assert "employment" in data
    assert "year" in data
    assert "period" in data


def test_combined_analysis(mock_world_bank, mock_bls):
    """Test combined industry analysis."""
    metrics = get_industry_analysis("5112")

    assert metrics.gdp_contribution > 0
    assert metrics.employment > 0
    assert metrics.growth_rate > 0
    assert metrics.confidence_score >= 0 and metrics.confidence_score <= 1
    assert len(metrics.sources) > 0


def test_cache_functionality():
    """Test caching functionality."""
    wb = WorldBankData()

    # First call should hit the API
    first_result = wb.get_industry_metrics()

    # Second call should use cache
    second_result = wb.get_industry_metrics()

    assert first_result == second_result  # Results should be identical


def test_error_handling():
    """Test error handling in market analysis."""
    estimator = MarketSizeEstimator([])  # Empty data

    # Should handle empty data gracefully
    result = estimator.estimate_market_size("test description", [])

    assert result.confidence_score < 0.5  # Low confidence due to missing data
