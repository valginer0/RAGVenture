import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_startups.core.rag_chain import extract_company_name, format_startup_idea

# Test cases
test_cases = [
    "Link helps fintech risk and AML compliance teams automate their workflows.",
    "Help desk software for modern teams.",
    "Deel helps companies hire anyone, anywhere.",
    "We build software for healthcare.",
    "The platform for all your needs.",
    "Stripe is a technology company that builds economic infrastructure.",
    "Link exists to help businesses grow.",
    "Platform that helps businesses scale.",
]

print("Testing extract_company_name():")
print("-" * 50)
for text in test_cases:
    company = extract_company_name(text)
    print(f"\nInput: {text}")
    print(
        f"Extracted company: {repr(company) if company else '[No company name found]'}"
    )

print("\n\nTesting full format_startup_idea():")
print("-" * 50)
for text in test_cases:
    result = format_startup_idea(text)
    print(f"\nInput: {text}")
    print(
        f"Company: {repr(result['Company']) if result['Company'] else '[No company name found]'}"
    )
    print(f"Problem: {result['Problem']}")
