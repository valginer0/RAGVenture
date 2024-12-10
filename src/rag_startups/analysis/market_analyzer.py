"""Market analysis functionality for startup ideas."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .external_data import (
    BLSData,
    IndustryMetrics,
    WorldBankData,
    get_industry_analysis,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketInsights:
    """Structured market analysis results."""

    market_size: float
    growth_rate: float
    competition_level: str
    barriers_to_entry: List[str]
    key_trends: List[str]
    risk_factors: List[str]
    opportunity_score: float  # 0-1 score
    confidence_score: float  # 0-1 score indicating confidence in the analysis
    year: int
    sources: List[str]


class MarketAnalyzer:
    """Analyzes market potential for startup ideas."""

    def __init__(self):
        """Initialize market analysis components."""
        self.world_bank = WorldBankData()
        self.bls = BLSData()

    def analyze_startup_idea(
        self, idea: Dict, country: str = "USA", year: Optional[int] = None
    ) -> MarketInsights:
        """
        Analyze market potential for a startup idea.

        Args:
            idea: Dictionary containing startup idea details
            country: Country code for market analysis
            year: Year for market data (default: latest available)

        Returns:
            MarketInsights object with analysis results
        """
        try:
            # Get basic industry metrics
            industry_code = self._detect_industry_code(idea["target_market"])
            metrics = get_industry_analysis(industry_code, country, year)

            # Analyze competition and trends
            competition_level = self._assess_competition(idea, metrics)
            barriers = self._identify_barriers(idea, metrics)
            trends = self._analyze_trends(idea, metrics)
            risks = self._assess_risks(idea, metrics)

            # Calculate opportunity score
            opportunity_score = self._calculate_opportunity_score(
                metrics, competition_level, barriers, trends, risks
            )

            return MarketInsights(
                market_size=metrics.market_size,
                growth_rate=metrics.growth_rate,
                competition_level=competition_level,
                barriers_to_entry=barriers,
                key_trends=trends,
                risk_factors=risks,
                opportunity_score=opportunity_score,
                confidence_score=metrics.confidence_score,
                year=metrics.year,
                sources=metrics.sources,
            )

        except Exception as e:
            logger.error(f"Failed to analyze idea {idea.get('name', 'Unknown')}: {e}")
            raise

    def _detect_industry_code(self, target_market: str) -> str:
        """Map target market description to industry code."""
        # TODO: Implement proper industry code detection
        # For now, return a default code
        return "5112"  # Software Publishers

    def _assess_competition(self, idea: Dict, metrics: IndustryMetrics) -> str:
        """Assess competition level based on market data."""
        # TODO: Implement proper competition assessment
        if metrics.market_size > 1_000_000_000:  # $1B
            return "High"
        elif metrics.market_size > 100_000_000:  # $100M
            return "Medium"
        return "Low"

    def _identify_barriers(self, idea: Dict, metrics: IndustryMetrics) -> List[str]:
        """Identify barriers to entry."""
        # TODO: Implement proper barriers analysis
        barriers = []
        if metrics.market_size > 1_000_000_000:
            barriers.append("High initial capital requirements")
        if "AI" in idea.get("solution", ""):
            barriers.append("Technical expertise required")
        return barriers or ["No significant barriers identified"]

    def _analyze_trends(self, idea: Dict, metrics: IndustryMetrics) -> List[str]:
        """Analyze market trends."""
        # TODO: Implement proper trends analysis
        trends = []
        if metrics.growth_rate > 10:
            trends.append("Rapid market growth")
        elif metrics.growth_rate > 5:
            trends.append("Steady market growth")
        else:
            trends.append("Mature market")
        return trends

    def _assess_risks(self, idea: Dict, metrics: IndustryMetrics) -> List[str]:
        """Assess potential risks."""
        # TODO: Implement proper risk assessment
        risks = []
        if metrics.growth_rate < 0:
            risks.append("Declining market")
        if metrics.market_size < 10_000_000:  # $10M
            risks.append("Small market size")
        return risks or ["No significant risks identified"]

    def _calculate_opportunity_score(
        self,
        metrics: IndustryMetrics,
        competition: str,
        barriers: List[str],
        trends: List[str],
        risks: List[str],
    ) -> float:
        """Calculate overall opportunity score (0-1)."""
        # TODO: Implement proper scoring
        score = 0.5  # Base score

        # Adjust for market size and growth
        if metrics.market_size > 1_000_000_000:
            score += 0.1
        if metrics.growth_rate > 10:
            score += 0.1

        # Adjust for competition
        if competition == "Low":
            score += 0.1
        elif competition == "High":
            score -= 0.1

        # Adjust for risks
        score -= len(risks) * 0.05

        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
