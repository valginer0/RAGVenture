"""Main entry point for the RAG Startups application.

This application combines RAG-based startup research with AI idea generation:
1. First, it finds relevant existing startups from the YC database
2. Then uses these as examples to generate new, innovative startup ideas

Required Environment Variables:
-----------------------------
For LangSmith integration:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="rag_startups"

For idea generation:
HUGGINGFACE_TOKEN="your_huggingface_token"

Optional Environment Variables:
----------------------------
LOCAL_LANGUAGE_MODEL: Name of the local language model to use (default: gpt2)
MAX_LINES: Maximum number of lines to process from the data file (default: 500000)
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict

from config.logging_config import setup_logging
from .core.rag_chain import calculate_result
from .utils.exceptions import RAGStartupsError
from .idea_generator.generator import StartupIdeaGenerator


def parse_startup_examples(rag_output: str) -> List[Dict]:
    """
    Parse RAG output to create example startups for the generator.
    Assumes RAG output contains startup descriptions in a structured format.
    """
    # TODO: Implement proper parsing based on RAG output format
    # For now, create a simple example from the RAG output
    return [{
        "name": "Example from YC",
        "problem": rag_output[:200],  # Use first 200 chars as problem description
        "solution": "Solution derived from YC example",
        "target_market": "Similar to YC startup",
        "unique_value": [
            "Based on successful YC startup",
            "Market-validated approach",
            "Proven business model"
        ]
    }]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Research existing startups and generate new ideas"
    )
    parser.add_argument(
        "industry",
        help="Industry or domain to research and generate ideas for (e.g., 'Real Estate', 'AI', 'Healthcare')",
    )
    parser.add_argument(
        "--data", 
        default="data/yc_startups.json", 
        help="Path to startup data JSON file"
    )
    parser.add_argument(
        "--model", 
        default="gpt2", 
        help="Name of the language model to use for RAG"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500_000,
        help="Maximum number of lines to process from YC data",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=3,
        help="Number of new ideas to generate (1-5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for idea generation (0.0-1.0)",
    )
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip YC research and generate ideas directly",
    )
    return parser.parse_args()


def main():
    """
    Main entry point that:
    1. Researches existing startups (using RAG)
    2. Uses findings to generate new ideas
    """
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Step 1: Research existing startups (unless skipped)
        example_startups = []
        if not args.skip_research:
            logger.info("Researching existing startups...")
            rag_result = calculate_result(
                args.industry,
                Path(args.data),
                args.model,
                args.max_lines,
            )
            print("\nRelevant Existing Startups:")
            print("-" * 40)
            print(rag_result)
            print("-" * 40)
            
            # Parse RAG results into examples
            example_startups = parse_startup_examples(rag_result)
        
        # Step 2: Generate new ideas based on research
        logger.info("Generating new startup ideas...")
        generator = StartupIdeaGenerator()
        
        # If no examples from research, create a basic example
        if not example_startups:
            example_startups = [{
                "name": "Industry Example",
                "problem": f"Various challenges in the {args.industry} sector",
                "solution": f"Innovative approaches to {args.industry} problems",
                "target_market": f"Businesses and consumers in {args.industry}",
                "unique_value": [
                    "Domain expertise",
                    "Novel solution",
                    "Market fit"
                ]
            }]
        
        ideas = generator.generate(
            num_ideas=args.num_ideas,
            example_startups=example_startups,
            temperature=args.temperature
        )
        
        if ideas:
            print("\nGenerated New Ideas:")
            print("-" * 40)
            print(ideas)
            print("-" * 40)
            return 0
        else:
            logger.error("No ideas were generated")
            return 1

    except RAGStartupsError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
