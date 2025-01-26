"""
Tests for the processors module.
"""

import re

import pytest

from rag_startups.idea_generator.processors import (
    generate_unique_name_suffix,
    parse_ideas,
)


def test_generate_unique_name_suffix():
    """Test that unique name suffixes are generated correctly."""
    # Test basic functionality
    name = "TechStartup"
    result = generate_unique_name_suffix(name)

    # Check format
    assert result.startswith(
        name + "-"
    ), "Result should start with original name and hyphen"
    assert len(result) == len(name) + 6, "Should add 5 char UUID and 1 hyphen"

    # Test uniqueness
    results = [generate_unique_name_suffix(name) for _ in range(100)]
    assert len(set(results)) == 100, "All generated names should be unique"


def test_parse_ideas_with_unique_names():
    """Test that parsed ideas have unique name suffixes."""
    sample_response = """
Startup Idea #1:
Name: TechCo
Problem/Opportunity: Problem 1
Solution: Solution 1
Target Market: Market 1

Startup Idea #2:
Name: DataCo
Problem/Opportunity: Problem 2
Solution: Solution 2
Target Market: Market 2
"""

    results = parse_ideas(sample_response)
    assert results is not None
    assert len(results) == 2

    # Check that names have the correct format
    for idea in results:
        name = idea["name"]
        # Should have original name + hyphen + 5 char UUID
        assert re.match(
            r"^[A-Za-z]+Co-[a-f0-9]{5}$", name
        ), f"Invalid name format: {name}"

    # Check that names are unique
    names = [idea["name"] for idea in results]
    assert len(set(names)) == len(names), "Generated names should be unique"
