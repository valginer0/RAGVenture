"""External data source integrations for market analysis."""

import datetime
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import requests
import wbdata
from dotenv import load_dotenv

# Load environment variables from both .env and system
load_dotenv(override=True)  # This will load .env and not override existing env vars

from ..utils.caching import cache_result

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
                    result = wbdata.get_data(indicator, country=country, date=str(year))
                    if result and len(result) > 0:
                        # Get the most recent value
                        data[indicator] = result[0]["value"]
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
        if self.api_key:
            logger.debug("BLS API key found in environment")
        else:
            logger.warning("BLS API key not found in environment")
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        self.series_mapping = {
            # NAICS 5112 - Software Publishers
            # CEU5051200001 - Software Publishers, All Employees, Thousands
            "5112": "CEU5051200001",
        }

    def register_api_key(self) -> str:
        """Get instructions for BLS API key registration."""
        return """
        To get a free BLS API key:
        1. Visit https://data.bls.gov/registrationEngine/
        2. Fill out the registration form
        3. Save the API key in your .env file as BLS_API_KEY=your_key_here
        """

    def get_employment_data(self, series_id: str) -> int:
        """Get employment data for a specific series ID."""
        try:
            response = self._make_request(series_id)
            if response and response.get("Results", {}).get("series"):
                series_data = response["Results"]["series"][0]["data"]
                if series_data:
                    # Convert thousands to actual number and round to nearest integer
                    latest_value = float(series_data[0]["value"]) * 1000
                    return int(round(latest_value))
            return 0
        except Exception as e:
            logger.error(f"Error fetching BLS data: {str(e)}")
            return 0

    def _make_request(self, series_id: str) -> Dict:
        headers = {"Content-type": "application/json"}
        data = {
            "seriesid": [series_id],
            "startyear": str(datetime.datetime.now().year - 1),
            "endyear": str(datetime.datetime.now().year),
            "registrationkey": self.api_key,
        }

        response = requests.post(self.base_url, json=data, headers=headers)
        response.raise_for_status()

        return response.json()

    @cache_result("bls", ttl=12 * 60 * 60)  # Cache for 12 hours
    def get_industry_employment_data(
        self,
        industry_code: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, Union[int, float]]:
        """Get employment data from BLS.

        Args:
            industry_code: Industry code (NAICS)
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

            # Get series ID from mapping
            series_id = self.series_mapping.get(industry_code)
            if not series_id:
                logger.warning(
                    f"No BLS series mapping found for industry code {industry_code}"
                )
                return {}

            logger.debug(f"Using BLS series ID: {series_id}")

            employment_data = self.get_employment_data(series_id)

            return {
                "employment": employment_data,
                "year": end_year,
                "period": "Annual",
            }
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

    # Get BLS employment data
    bls_data = bls.get_industry_employment_data(industry_code)

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
