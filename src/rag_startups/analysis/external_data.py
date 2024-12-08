"""External data source integrations for market analysis."""

import datetime
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import requests
import wbdata
from dotenv import load_dotenv

from ..utils.caching import cache_result

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class IndustryMetrics:
    """Industry metrics from external sources."""

    gdp_contribution: float  # in USD
    employment: int
    growth_rate: float  # yearly growth rate
    market_size: float  # estimated market size in USD
    confidence_score: float  # 0.0 to 1.0
    year: int
    sources: List[str]


class WorldBankData:
    """World Bank data integration."""

    def __init__(self):
        """Initialize World Bank data client."""
        self.indicators = {
            "NY.GDP.MKTP.CD": "GDP (current US$)",
            "NV.IND.TOTL.ZS": "Industry (% of GDP)",
            "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
        }

    @cache_result("worldbank", ttl=24 * 60 * 60)  # Cache for 24 hours
    def get_industry_metrics(
        self, country: str = "USA", year: Optional[int] = None
    ) -> Dict[str, float]:
        """Get industry metrics from World Bank.

        Args:
            country: Country code (default: USA)
            year: Year for data (default: latest available)

        Returns:
            Dictionary of industry metrics
        """
        try:
            if year is None:
                year = datetime.datetime.now().year - 1  # Previous year

            # Get data for each indicator separately and combine
            data = {}
            for indicator in self.indicators:
                try:
                    # World Bank API expects dates in format "YYYY"
                    result = wbdata.get_data(
                        indicator, 
                        country=country, 
                        date=str(year)
                    )
                    if result and len(result) > 0:
                        # Get the most recent value
                        data[indicator] = result[0]['value']
                except Exception as e:
                    logger.warning(f"Failed to fetch indicator {indicator}: {e}")
                    data[indicator] = 0

            return {
                "gdp": data.get("NY.GDP.MKTP.CD", 0),
                "industry_percentage": data.get("NV.IND.TOTL.ZS", 0),
                "growth_rate": data.get("NY.GDP.MKTP.KD.ZG", 0),
            }
        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            return {}


class BLSData:
    """Bureau of Labor Statistics data integration."""

    def __init__(self):
        """Initialize BLS client with API key."""
        self.api_key = os.getenv("BLS_API_KEY")
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def register_api_key(self) -> str:
        """Get instructions for BLS API key registration."""
        return """
        To get a free BLS API key:
        1. Visit https://data.bls.gov/registrationEngine/
        2. Fill out the registration form
        3. Save the API key in your .env file as BLS_API_KEY=your_key_here
        """

    @cache_result("bls", ttl=12 * 60 * 60)  # Cache for 12 hours
    def get_employment_data(
        self,
        series_id: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, Union[int, float]]:
        """Get employment data from BLS.

        Args:
            series_id: BLS series identifier
            start_year: Start year for data (default: previous year)
            end_year: End year for data (default: current year)

        Returns:
            Dictionary with employment data
        """
        if not self.api_key:
            logger.warning(
                "BLS API key not found. Use register_api_key() for instructions."
            )
            return {}

        try:
            if start_year is None:
                start_year = datetime.datetime.now().year - 1
            if end_year is None:
                end_year = datetime.datetime.now().year

            headers = {"Content-type": "application/json"}
            data = {
                "seriesid": [series_id],
                "startyear": str(start_year),
                "endyear": str(end_year),
                "registrationkey": self.api_key,
            }

            response = requests.post(self.base_url, json=data, headers=headers)
            response.raise_for_status()

            results = response.json()
            if results.get("status") == "REQUEST_SUCCEEDED":
                series_data = results["Results"]["series"][0]["data"]
                return {
                    "employment": int(series_data[0]["value"]),
                    "year": int(series_data[0]["year"]),
                    "period": series_data[0]["period"],
                }
            else:
                logger.error(f"BLS API error: {results.get('message')}")
                return {}

        except Exception as e:
            logger.error(f"Error fetching BLS data: {e}")
            return {}


def get_industry_analysis(
    industry_code: str, country: str = "USA", year: Optional[int] = None
) -> IndustryMetrics:
    """Get comprehensive industry analysis using multiple data sources.

    Args:
        industry_code: Industry classification code
        country: Country code (default: USA)
        year: Year for analysis (default: latest available)

    Returns:
        IndustryMetrics object with combined analysis
    """
    # Initialize data sources
    wb = WorldBankData()
    bls = BLSData()

    # Get World Bank data
    wb_data = wb.get_industry_metrics(country, year)

    # Get BLS data if available
    bls_data = bls.get_employment_data(f"CEU{industry_code}00000001")

    # Calculate market size
    gdp = wb_data.get("gdp", 0)
    industry_pct = wb_data.get("industry_percentage", 0)
    market_size = (gdp * industry_pct / 100) if gdp and industry_pct else 0

    # Calculate confidence score based on data availability
    confidence_factors = [bool(wb_data), bool(bls_data), bool(market_size > 0)]
    confidence_score = sum(confidence_factors) / len(confidence_factors)

    return IndustryMetrics(
        gdp_contribution=market_size,
        employment=bls_data.get("employment", 0),
        growth_rate=wb_data.get("growth_rate", 0),
        market_size=market_size,
        confidence_score=confidence_score,
        year=year or datetime.datetime.now().year - 1,
        sources=["World Bank", "BLS"] if bls_data else ["World Bank"],
    )
