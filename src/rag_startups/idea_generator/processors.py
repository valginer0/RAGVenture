"""
Response processing utilities for startup idea generation.
"""

import re
from typing import Optional


def standardize_field_names(text: str) -> str:
    """
    Standardize field names in the response to have consistent capitalization.

    Args:
        text: Raw response text

    Returns:
        Text with standardized field names
    """
    fields = [
        "name",
        "problem/opportunity",
        "solution",
        "target market",
        "unique value",
    ]

    # For each field, find any case variation followed by ':' and replace with proper capitalization
    for field in fields:
        pattern = re.compile(f"{field}:", re.IGNORECASE)
        text = pattern.sub(f"{field.title()}:", text)

    return text


def clean_response(response: str) -> Optional[str]:
    """
    Clean and format the response from the model.

    Args:
        response: Raw response from the model

    Returns:
        Cleaned response text or None if invalid
    """
    try:
        # Extract only the generated ideas after the prompt
        response_text = response.strip()
        if "BEGIN YOUR RESPONSE NOW:" in response_text:
            ideas_only = response_text.split("BEGIN YOUR RESPONSE NOW:")[1].strip()
        else:
            ideas_only = response_text

        # Clean up the response
        ideas_only = ideas_only.replace("\n---\n", "\n\n")  # Remove markdown separators
        ideas_only = standardize_field_names(ideas_only)

        return ideas_only
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return None


def parse_ideas(cleaned_response: str) -> list[dict]:
    """
    Parse cleaned response into structured data.

    Args:
        cleaned_response: Cleaned response text

    Returns:
        List of dictionaries containing structured startup ideas
    """
    # TODO: Implement parsing logic to convert text into structured data
    # This will be useful for storing in database or further processing
    pass
