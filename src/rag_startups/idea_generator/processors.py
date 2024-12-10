"""
Response processing utilities for startup idea generation.
"""

import re
from typing import Dict, List, Optional


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


def parse_ideas(cleaned_response: str) -> Optional[List[Dict[str, str]]]:
    """
    Parse cleaned response into structured data.

    Args:
        cleaned_response: Cleaned response text

    Returns:
        List of dictionaries containing structured startup ideas or None if parsing fails
    """
    if not cleaned_response:
        return None

    try:
        # Split into individual ideas
        ideas_raw = re.split(r"Startup Idea #?\d+:?", cleaned_response)
        ideas_raw = [idea.strip() for idea in ideas_raw if idea.strip()]

        parsed_ideas = []
        for idea_text in ideas_raw:
            idea_dict = {}

            # Extract fields using regex
            name_match = re.search(r"Name:\s*(.+?)(?:\n|$)", idea_text)
            problem_match = re.search(
                r"Problem/Opportunity:\s*(.+?)(?:\n|$)", idea_text
            )
            solution_match = re.search(r"Solution:\s*(.+?)(?:\n|$)", idea_text)
            market_match = re.search(r"Target Market:\s*(.+?)(?:\n|$)", idea_text)

            # Only add idea if we have the minimum required fields
            if name_match and problem_match and solution_match and market_match:
                idea_dict["name"] = name_match.group(1).strip()
                idea_dict["problem"] = problem_match.group(1).strip()
                idea_dict["solution"] = solution_match.group(1).strip()
                idea_dict["target_market"] = market_match.group(1).strip()

                # Extract unique value points if present
                value_match = re.search(
                    r"Unique Value:(.+?)(?=\n\w+:|$)", idea_text, re.DOTALL
                )
                if value_match:
                    values = value_match.group(1).strip()
                    # Convert bullet points to list
                    values = [
                        v.strip().lstrip("•").strip()
                        for v in values.split("\n")
                        if v.strip()
                    ]
                    idea_dict["unique_value"] = values

                parsed_ideas.append(idea_dict)

        return parsed_ideas if parsed_ideas else None

    except Exception as e:
        print(f"Error parsing ideas: {str(e)}")
        return None
