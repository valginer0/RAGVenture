"""Tests for market analysis functionality."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from rag_startups.analysis.external_data import (
    BLSData,
    IndustryMatch,
    IndustryMetrics,
    MarketRelationship,
    MarketRelationType,
    MultiMarketInsights,
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
def mock_world_bank(monkeypatch):
    """Mock World Bank API responses."""

    def mock_wb_metrics(self, country="USA", year=None):
        return {
            "gdp": 20000000000000,  # $20T
            "industry_percentage": 20,  # 20% of GDP
            "growth_rate": 2.5,
        }

    monkeypatch.setattr(WorldBankData, "get_industry_metrics", mock_wb_metrics)


@pytest.fixture
def mock_bls(monkeypatch):
    """Mock BLS API responses."""

    def mock_make_request(self, series_id):
        # Map series IDs to employment values
        employment_data = {
            "CEU5552110001": 2000,  # Banking (5221)
            "CEU6562110001": 3000,  # Healthcare (6211)
            "CEU5051200001": 1500,  # Software (5112)
            "CEU5051820001": 1000,  # Cloud (5182)
        }

        if series_id not in employment_data:
            return {"status": "REQUEST_FAILED", "Results": None}

        return {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "data": [
                            {
                                "year": "2023",
                                "period": "M12",
                                "value": str(employment_data[series_id]),
                                "periodName": "December",
                                "latest": True,
                            }
                        ]
                    }
                ]
            },
        }

    def mock_wb_metrics(self, country="USA", year=None):
        return {
            "gdp": 20000000000000,  # $20T
            "industry_percentage": 20,  # 20% of GDP
            "growth_rate": 2.5,
        }

    monkeypatch.setattr(BLSData, "_make_request", mock_make_request)
    monkeypatch.setattr(WorldBankData, "get_industry_metrics", mock_wb_metrics)


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
    data = bls.get_employment_data(
        "CEU5051200001"
    )  # Series ID for Software Publishers (5112)

    assert "employment" in data
    assert data["employment"] > 0


def test_combined_analysis(mock_world_bank, mock_bls):
    """Test combined industry analysis."""
    metrics = get_industry_analysis("Software development and publishing platform")

    assert metrics.primary_market.gdp_contribution > 0
    assert metrics.primary_market.employment > 0
    assert metrics.primary_market.growth_rate > 0
    assert metrics.primary_market.market_size > 0
    assert 0 <= metrics.primary_market.confidence_score <= 1
    assert metrics.primary_market.year > 0
    assert len(metrics.primary_market.sources) > 0

    # Test combined metrics
    assert metrics.combined_market_size > 0
    assert metrics.combined_growth_rate > 0
    assert 0 <= metrics.confidence_score <= 1


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


try:
    import spacy

    spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy model not available")
def test_industry_code_detection():
    """Test the industry code detection functionality."""
    bls = BLSData()

    test_cases = [
        # Technology and Software
        (
            "A SaaS platform for enterprise resource planning",
            "5112",
        ),  # Software Publishers
        (
            "IT consulting and systems integration services",
            "5415",
        ),  # Computer Systems Design
        (
            "Cloud hosting and data center services",
            "5182",
        ),  # Data Processing and Hosting
        # E-commerce and Retail
        ("Online marketplace for handmade products", "4510"),  # E-commerce
        ("Digital-first retail technology platform", "4529"),  # General Merchandise
        # Healthcare
        (
            "Telehealth platform for remote medical consultations",
            "6211",
        ),  # Medical Offices
        ("AI-powered medical diagnostics laboratory", "6215"),  # Medical Labs
        ("Biotech startup developing new therapeutics", "3254"),  # Pharmaceuticals
        # Financial Services
        ("Digital banking platform for millennials", "5221"),  # Banking
        ("AI-powered investment management platform", "5239"),  # Financial Investment
        ("Digital insurance marketplace", "5242"),  # Insurance
        # Hardware and Manufacturing
        ("Consumer electronics and smart home devices", "3341"),  # Computer Hardware
        ("Semiconductor chip design and manufacturing", "3344"),  # Semiconductors
        # Media and Entertainment
        ("Video streaming platform for educational content", "5121"),  # Motion Picture
        ("Podcast hosting and distribution platform", "5122"),  # Sound Recording
        # Logistics
        ("Last-mile delivery automation platform", "4921"),  # Couriers
        ("Smart warehousing and fulfillment solutions", "4931"),  # Warehousing
    ]

    for description, expected_code in test_cases:
        matches = bls._detect_industry_codes(description)
        assert len(matches) >= 1, f"No matches found for: {description}"
        detected_code = matches[0].code
        assert detected_code == expected_code, (
            f"Failed to detect correct industry code for: {description}\n"
            f"Expected: {expected_code}, Got: {detected_code}"
        )


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy model not available")
def test_industry_code_detection_multi():
    bls = BLSData()

    test_cases = [
        # Single industry
        (
            "A software company that makes cloud-based applications",
            [("5112", 0.5)],  # Software Publishers
        ),
        # Healthcare fintech
        (
            "A healthcare payment processing platform for hospitals",
            [("5221", 0.3), ("6211", 0.3)],  # Banking + Healthcare
        ),
        # Insurtech
        (
            "An AI-powered health insurance platform",
            [("5242", 0.3), ("6211", 0.3)],  # Insurance + Healthcare
        ),
        # Software + Cloud
        (
            "A cloud-based software development platform",
            [("5112", 0.3), ("5182", 0.3)],  # Software + Cloud
        ),
    ]

    for description, expected_matches in test_cases:
        matches = bls._detect_industry_codes(description)
        assert len(matches) >= len(expected_matches), (
            f"Not enough matches for: {description}\n"
            f"Expected at least {len(expected_matches)} matches, got {len(matches)}"
        )

        detected_codes = {m.code: m.confidence for m in matches}
        for code, min_confidence in expected_matches:
            assert code in detected_codes, (
                f"Missing expected code {code} for: {description}\n"
                f"Detected codes: {list(detected_codes.keys())}"
            )
            assert detected_codes[code] >= min_confidence, (
                f"Low confidence for {code} in: {description}\n"
                f"Expected >= {min_confidence}, got {detected_codes[code]}"
            )


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy model not available")
def test_multi_industry_detection():
    bls = BLSData()

    # Test healthcare fintech
    matches = bls._detect_industry_codes(
        "Healthcare payment platform for hospitals and insurance companies"
    )
    codes = [m.code for m in matches]
    assert "6211" in codes  # Healthcare
    assert "5221" in codes  # Banking/Fintech
    assert "5242" in codes  # Insurance

    # Test e-commerce platform
    matches = bls._detect_industry_codes(
        "Online marketplace platform for retail merchants"
    )
    codes = [m.code for m in matches]
    assert "4510" in codes  # E-commerce
    assert "5112" in codes  # Software

    # Test pure fintech
    matches = bls._detect_industry_codes(
        "Digital banking and payment processing platform"
    )
    codes = [m.code for m in matches]
    assert "5221" in codes  # Banking/Fintech
    assert matches[0].code == "5221"  # Should be primary match

    # Test insurtech
    matches = bls._detect_industry_codes("AI-powered insurance underwriting platform")
    codes = [m.code for m in matches]
    assert "5242" in codes  # Insurance
    assert matches[0].code == "5242"  # Should be primary match
    assert matches[0].code == "5242"  # Should be primary match


def test_market_relationship_calculation():
    bls = BLSData()

    # Test independent markets (healthcare + fintech)
    markets = [
        IndustryMetrics(
            industry_code="5221",
            gdp_contribution=1000,
            employment=50000,
            growth_rate=0.05,
            market_size=1000,
            confidence_score=0.8,
            year=2023,
            sources=["Test"],
        ),
        IndustryMetrics(
            industry_code="6211",
            gdp_contribution=800,
            employment=40000,
            growth_rate=0.03,
            market_size=800,
            confidence_score=0.7,
            year=2023,
            sources=["Test"],
        ),
    ]
    combined = bls._calculate_combined_metrics(markets, [])

    # With default 10% overlap
    expected_size = 1000 + (800 * 0.9)  # Second market minus 10% overlap
    assert abs(combined["market_size"] - expected_size) < 0.01
    assert abs(combined["growth_rate"] - 0.04) < 0.01  # Average of 5% and 3%


def test_get_industry_analysis_multi():
    """Test multi-market analysis for healthcare fintech."""
    # Create a mock BLS instance
    bls = BLSData()

    # Create test metrics
    banking_metrics = IndustryMetrics(
        industry_code="5221",
        gdp_contribution=1000,
        employment=50000,
        growth_rate=0.05,
        market_size=1000,
        confidence_score=0.8,
        year=2023,
        sources=["Test"],
    )

    healthcare_metrics = IndustryMetrics(
        industry_code="6211",
        gdp_contribution=800,
        employment=40000,
        growth_rate=0.03,
        market_size=800,
        confidence_score=0.7,
        year=2023,
        sources=["Test"],
    )

    # Override market relationships for testing
    relationship = MarketRelationship(
        industry1="5221",
        industry2="6211",
        relationship=MarketRelationType.INDEPENDENT,
        overlap_factor=0.2,  # 20% overlap in healthcare fintech
    )
    bls.market_relationships = {
        ("5221", "6211"): relationship,
        ("6211", "5221"): relationship,  # Add reverse relationship
    }

    # Mock industry detection to return healthcare and fintech
    def mock_detect_codes(*args, **kwargs):
        return [
            IndustryMatch(
                code="5221",
                score=0.9,
                confidence=0.8,
                keywords_matched=["payment", "financial"],
            ),
            IndustryMatch(
                code="6211", score=0.8, confidence=0.7, keywords_matched=["healthcare"]
            ),
        ]

    bls._detect_industry_codes = mock_detect_codes

    # Mock get_industry_metrics to return test data
    def mock_get_metrics(code, year=None):
        metrics = {"5221": banking_metrics, "6211": healthcare_metrics}
        return metrics.get(code)

    bls.get_industry_metrics = mock_get_metrics

    # Create a mock WorldBankData instance
    wb = WorldBankData()

    def mock_wb_metrics(*args, **kwargs):
        return {
            "gdp": 20000000000000,  # $20T
            "industry_percentage": 20,  # 20% of GDP
            "growth_rate": 2.5,
        }

    wb.get_industry_metrics = mock_wb_metrics

    # Mock the BLSData constructor to return our mocked instance
    with patch("rag_startups.analysis.external_data.BLSData", return_value=bls):
        with patch(
            "rag_startups.analysis.external_data.WorldBankData", return_value=wb
        ):
            insights = get_industry_analysis("A healthcare payment processing platform")

            assert isinstance(insights, MultiMarketInsights)
            assert insights.primary_market is not None
            assert len(insights.related_markets) >= 1
            assert insights.primary_market.industry_code in ["5221", "6211"]

            # Verify relationships are properly set
            assert len(insights.relationships) >= 1
            first_relationship = insights.relationships[0]
            assert isinstance(first_relationship, MarketRelationship)
            assert first_relationship.overlap_factor == 0.2
