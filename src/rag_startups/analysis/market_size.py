"""Market size estimation module for startup analysis."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .external_data import IndustryMetrics, get_industry_analysis


class MarketSegment(Enum):
    """Market segment categories."""

    B2B = "b2b"
    B2C = "b2c"
    ENTERPRISE = "enterprise"
    SMB = "smb"
    UNKNOWN = "unknown"


class MarketStage(Enum):
    """Market maturity stages."""

    EMERGING = "emerging"
    GROWING = "growing"
    MATURE = "mature"
    DECLINING = "declining"
    UNKNOWN = "unknown"


@dataclass
class MarketSize:
    """Market size estimation results."""

    total_addressable_market: float  # in billions USD
    serviceable_addressable_market: float
    serviceable_obtainable_market: float
    segment: MarketSegment
    stage: MarketStage
    year: int
    confidence_score: float
    sources: List[str]


class MarketSizeEstimator:
    """Estimates market size for startup ideas."""

    def __init__(self, startup_data: List[Dict]):
        """Initialize with historical startup data.

        Args:
            startup_data: List of startup dictionaries containing metadata
        """
        self.startup_data = startup_data
        self._segment_patterns = self._compile_segment_patterns()

    def _compile_segment_patterns(self) -> Dict[MarketSegment, List[str]]:
        """Compile regex patterns for market segmentation."""
        return {
            MarketSegment.B2B: [
                r"b2b",
                r"business[\s-]to[\s-]business",
                r"enterprise\s+software",
                r"saas",
            ],
            MarketSegment.B2C: [
                r"b2c",
                r"business[\s-]to[\s-]consumer",
                r"consumer",
                r"retail",
                r"mobile\s+app",
                r"end[\s-]user",
            ],
            MarketSegment.ENTERPRISE: [
                r"enterprise",
                r"large\s+business",
                r"corporate",
            ],
            MarketSegment.SMB: [r"smb", r"small\s+business", r"medium\s+business"],
        }

    def estimate_market_size(
        self, description: str, similar_startups: List[Dict], year: int = 2024
    ) -> MarketSize:
        """Estimate market size for a startup idea.

        Args:
            description: Startup idea description
            similar_startups: List of similar startups from the database
            year: Target year for estimation (default: current year)

        Returns:
            MarketSize object containing the estimation results
        """
        # Determine market segment
        segment = self._determine_segment(description)

        # Calculate market sizes based on similar startups and patterns
        tam = self._estimate_total_addressable_market(description, similar_startups)
        sam = tam * 0.4  # Typically 30-50% of TAM
        som = sam * 0.1  # Typically 10-20% of SAM

        # Determine market stage
        stage = self._determine_market_stage(similar_startups)

        # Calculate confidence score based on data quality
        confidence = self._calculate_confidence_score(similar_startups)

        return MarketSize(
            total_addressable_market=tam,
            serviceable_addressable_market=sam,
            serviceable_obtainable_market=som,
            segment=segment,
            stage=stage,
            year=year,
            confidence_score=confidence,
            sources=["Historical startup data", "Pattern analysis"],
        )

    def _determine_segment(self, description: str) -> MarketSegment:
        """Determine market segment based on description patterns."""
        description = description.lower()

        for segment, patterns in self._segment_patterns.items():
            if any(re.search(pattern, description) for pattern in patterns):
                return segment

        return MarketSegment.UNKNOWN

    def _estimate_total_addressable_market(
        self, description: str, similar_startups: List[Dict]
    ) -> float:
        """Estimate total addressable market size in billions USD."""
        # Get industry code based on description
        industry_code = self._get_industry_code(description)

        # Get external market data
        industry_metrics = get_industry_analysis(industry_code)

        # Base estimate from similar startups
        startup_based_estimate = (
            sum(s.get("valuation", 0) for s in similar_startups) / len(similar_startups)
            if similar_startups
            else 0
        )

        # Combine external data with startup-based estimate
        if industry_metrics.market_size > 0:
            # Weight external data more heavily if confidence is high
            weight = industry_metrics.confidence_score
            combined_estimate = (
                weight * industry_metrics.market_size
                + (1 - weight) * startup_based_estimate * 20  # Market multiplier
            )
            return combined_estimate / 1e9  # Convert to billions
        else:
            # Fall back to startup-based estimate if no external data
            return startup_based_estimate * 20 / 1e9

    def _get_industry_code(self, description: str) -> str:
        """Get BLS industry code from description."""
        # TODO: Implement more sophisticated industry code mapping
        # For now, return a default code for software industry
        return "5112"  # Software Publishers

    def _determine_market_stage(self, similar_startups: List[Dict]) -> MarketStage:
        """Determine market stage based on similar startups."""
        # TODO: Implement stage determination using:
        # 1. Growth rates
        # 2. Number of competitors
        # 3. Investment patterns
        return MarketStage.GROWING  # Default for now

    def _calculate_confidence_score(self, similar_startups: List[Dict]) -> float:
        """Calculate confidence score for the estimation."""
        # Factors affecting confidence:
        # 1. Number of similar startups
        # 2. Data completeness
        # 3. Market similarity
        base_score = min(len(similar_startups) / 10, 1.0)  # 0.0 to 1.0
        return base_score * 0.8  # Conservative adjustment
