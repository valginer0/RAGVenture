#!/usr/bin/env python3
"""Demo script for market analysis features."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.rag_startups.analysis.external_data import get_industry_analysis
from src.rag_startups.analysis.market_size import MarketSizeEstimator
from src.rag_startups.data.loader import load_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze market potential for startups"
    )

    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Description of the startup idea (e.g., 'B2B SaaS platform for healthcare')",
    )
    parser.add_argument(
        "--industry-code",
        type=str,
        default="5112",  # Software Publishers
        help="Industry code for analysis (default: 5112 - Software Publishers)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="yc_startups.json",
        help="Path to the JSON file containing startup data (default: yc_startups.json)",
    )

    return parser.parse_args()


def main():
    """Run market analysis demo."""
    args = parse_arguments()

    # Initialize market size estimator with sample data
    startup_data = [
        {"name": "Fintech App A", "valuation": 500000000},  # $500M
        {"name": "Personal Finance B", "valuation": 750000000},  # $750M
        {"name": "Mobile Banking C", "valuation": 1200000000},  # $1.2B
    ]

    print("\nLoading startup data...")
    estimator = MarketSizeEstimator(startup_data)

    # Get market size estimation
    print(f"\nAnalyzing market size for: {args.description}")
    market_size = estimator.estimate_market_size(
        description=args.description,
        similar_startups=startup_data,  # Pass sample data
        year=2024,
    )

    print("\nMarket Size Analysis Results:")
    print(
        f"Total Addressable Market (TAM): ${market_size.total_addressable_market:.2f}B"
    )
    print(
        f"Serviceable Addressable Market (SAM): ${market_size.serviceable_addressable_market:.2f}B"
    )
    print(
        f"Serviceable Obtainable Market (SOM): ${market_size.serviceable_obtainable_market:.2f}B"
    )
    print(f"Market Segment: {market_size.segment.value}")
    print(f"Market Stage: {market_size.stage.value}")
    print(f"Confidence Score: {market_size.confidence_score:.2f}")

    # Get industry analysis
    print(f"\nFetching industry analysis for code: {args.industry_code}")
    metrics = get_industry_analysis(args.industry_code)

    print("\nIndustry Analysis Results:")
    print(f"GDP Contribution: ${metrics.gdp_contribution/1e9:.2f}B")
    print(f"Employment: {metrics.employment:,}")
    print(f"Growth Rate: {metrics.growth_rate:.1f}%")
    print(f"Market Size: ${metrics.market_size/1e9:.2f}B")
    print(f"Confidence Score: {metrics.confidence_score:.2f}")
    print(f"Data Sources: {', '.join(metrics.sources)}")


if __name__ == "__main__":
    main()
