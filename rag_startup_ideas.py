"""
To use langsmith set up the enviromnent variables :

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your lansmith api key"
LANGCHAIN_PROJECT="the name of your langsmith project"

"""

import argparse
import os
from pathlib import Path

from embed_master import calculate_result, initialize_embeddings
from src.rag_startups.core.startup_metadata import StartupLookup
from src.rag_startups.data.loader import load_data
from src.rag_startups.utils.output_formatter import formatter


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate startup ideas using RAG")

    # Required arguments
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic or domain to generate startup ideas for (e.g., 'healthcare', 'education technology')",
    )

    # Optional arguments
    parser.add_argument(
        "--file",
        type=str,
        default="yc_startups.json",
        help="Path to the JSON file containing startup data (default: yc_startups.json)",
    )
    parser.add_argument(
        "--max-lines", type=int, default=None, help="Maximum number of lines to process"
    )
    parser.add_argument(
        "--num-ideas", type=int, default=3, help="Number of startup ideas to generate"
    )
    parser.add_argument(
        "--market-analysis",
        action="store_true",
        help="Enable market analysis for generated ideas",
    )

    # Future extensibility (commented out for now)
    # parser.add_argument('--chunk-size', type=int, default=1000,
    #                    help='Size of text chunks for processing')
    # parser.add_argument('--chunk-overlap', type=int, default=200,
    #                    help='Overlap between text chunks')
    # parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
    #                    help='Model to use for embeddings')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        print(
            f"Make sure the file exists or specify a different file with --file option."
        )
        exit(1)

    # Construct a natural query based on the topic
    topic = args.topic.strip().lower()
    question = (
        f"Find innovative startup ideas in {topic}"
        if " " in topic  # If it's a compound phrase like "education technology"
        else f"Find innovative startup ideas in the {topic} domain"
    )

    prompt_messages = [
        (
            "system",
            """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            Answer:""",
        )
    ]

    # Load data once and get both DataFrame and JSON data
    df, json_data = load_data(args.file, args.max_lines)

    # Initialize lookup with the JSON data
    lookup = StartupLookup(json_data)

    # Initialize embeddings and retriever once
    retriever = initialize_embeddings(df)

    # Pass retriever to calculate_result
    result = calculate_result(
        question=question,
        retriever=retriever,
        json_data=json_data,
        prompt_messages=prompt_messages,
        lookup=lookup,
        num_ideas=args.num_ideas,
        include_market_analysis=args.market_analysis,
    )

    # Print the results in a nicely formatted way
    formatter.print_startup_ideas(result)
    formatter.print_summary()
