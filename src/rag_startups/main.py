"""Main entry point for the RAG Startups application.

Required Environment Variables:
-----------------------------
To use LangSmith integration, set up the following environment variables:

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="rag_startups"

Optional Environment Variables:
----------------------------
LOCAL_LANGUAGE_MODEL: Name of the local language model to use (default: gpt2)
MAX_LINES: Maximum number of lines to process from the data file (default: 500000)
"""
import argparse
import logging
import sys
from pathlib import Path

from config.logging_config import setup_logging
from .core.rag_chain import calculate_result
from .utils.exceptions import RAGStartupsError

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find relevant startup ideas from YC database"
    )
    parser.add_argument(
        "industry",
        help="Industry or domain to find startup ideas for (e.g., 'Real Estate', 'AI', 'Healthcare')"
    )
    parser.add_argument(
        "--data",
        default="data/yc_startups.json",
        help="Path to startup data JSON file"
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Name of the language model to use"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500_000,
        help="Maximum number of lines to process"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Ensure data file exists
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Generate result
        result = calculate_result(
            question=args.industry,
            file_path=str(data_path),
            prompt_messages=[],  
            model_name=args.model,
            max_lines=args.max_lines
        )
        
        print("\nRelevant Startup Ideas:")
        print("----------------------------------------")
        print(result)
        print("----------------------------------------")
        
        return 0
        
    except RAGStartupsError as e:
        logger.error(f"Application error: {e}")
        return 1
    except Exception as e:
        logger.exception("Unexpected error")
        return 1

if __name__ == "__main__":
    sys.exit(main())
