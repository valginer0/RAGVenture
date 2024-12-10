"""Tests for market analysis functionality."""

from unittest.mock import Mock, patch

import pytest

from rag_startups.analysis.external_data import IndustryMetrics
from rag_startups.analysis.market_analyzer import MarketAnalyzer, MarketInsights


@pytest.fixture
def sample_idea():
    return {
        "name": "TechStartup",
        "problem": "Complex data analysis is hard",
        "solution": "AI-powered analytics platform",
        "target_market": "Enterprise software companies",
        "unique_value": ["Easy to use", "AI-powered", "Cost-effective"],
    }


@pytest.fixture
def sample_metrics():
    return IndustryMetrics(
        gdp_contribution=1_000_000_000,  # $1B
        employment=50000,
        growth_rate=15.0,
        market_size=5_000_000_000,  # $5B
        confidence_score=0.8,
        year=2023,
        sources=["World Bank", "BLS"],
    )


def test_market_analyzer_initialization():
    """Test that MarketAnalyzer initializes correctly."""
    analyzer = MarketAnalyzer()
    assert analyzer.world_bank is not None
    assert analyzer.bls is not None


@patch("rag_startups.analysis.market_analyzer.get_industry_analysis")
def test_analyze_startup_idea(mock_get_industry, sample_idea, sample_metrics):
    """Test basic market analysis functionality."""
    mock_get_industry.return_value = sample_metrics

    analyzer = MarketAnalyzer()
    result = analyzer.analyze_startup_idea(sample_idea)

    assert isinstance(result, MarketInsights)
    assert result.market_size == sample_metrics.market_size
    assert result.growth_rate == sample_metrics.growth_rate
    assert isinstance(result.competition_level, str)
    assert isinstance(result.barriers_to_entry, list)
    assert isinstance(result.key_trends, list)
    assert isinstance(result.risk_factors, list)
    assert 0 <= result.opportunity_score <= 1
    assert result.year == sample_metrics.year
    assert result.sources == sample_metrics.sources


def test_opportunity_score_calculation(sample_metrics):
    """Test that opportunity score is calculated correctly."""
    analyzer = MarketAnalyzer()
    score = analyzer._calculate_opportunity_score(
        metrics=sample_metrics,
        competition="Low",
        barriers=["High initial capital"],
        trends=["Rapid market growth"],
        risks=["Competitive market"],
    )

    assert 0 <= score <= 1


def test_competition_assessment(sample_idea, sample_metrics):
    """Test competition level assessment."""
    analyzer = MarketAnalyzer()
    competition = analyzer._assess_competition(sample_idea, sample_metrics)
    assert competition in ["Low", "Medium", "High"]


def test_error_handling():
    """Test error handling in market analysis."""
    analyzer = MarketAnalyzer()
    with pytest.raises(Exception):
        analyzer.analyze_startup_idea({})  # Empty idea should raise error
