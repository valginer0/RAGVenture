from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


@dataclass
class MarketRelationType(Enum):
    INDEPENDENT = "independent"
    COMPLEMENTARY = "complementary"
    SUBSTITUTE = "substitute"


@dataclass
class MarketRelationship:
    industry1: str
    industry2: str
    relationship: MarketRelationType
    overlap_factor: float


@dataclass
class MultiMarketInsights:
    primary_market: Optional["IndustryMetrics"]
    related_markets: List["IndustryMetrics"]
    relationships: List[MarketRelationship]
    combined_market_size: float
    combined_growth_rate: float
    confidence_score: float
    year: int
    sources: List[str]
