#!/usr/bin/env python3
"""Quick test for GRI pattern detection"""

import re

def test_organization_patterns(text: str):
    """Test the organization pattern detection"""
    patterns = []

    # Pattern: "Full Organization Name (ACRONYM)" - be more precise
    pattern = r'(?:^|(?<=\s))([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*\(([A-Z]{2,6})\)'

    print(f"Testing text: {text}")
    print(f"Using regex pattern: {pattern}")
    print()

    for match in re.finditer(pattern, text):
        full_name = match.group(1).strip()
        acronym = match.group(2).strip()
        start, end = match.span()

        print(f"Raw match: '{match.group(0)}'")
        print(f"  Full name: '{full_name}'")
        print(f"  Acronym: '{acronym}'")
        print(f"  Position: {start}-{end}")
        print(f"  Full name word count: {len(full_name.split())}")
        print(f"  Acronym length: {len(acronym)}")

        # Check if this looks like an organization
        if (len(full_name.split()) >= 2 and  # Multi-word
            not any(word.lower() in ['the', 'of', 'and', 'for', 'in'] for word in full_name.split()[0:1]) and  # Doesn't start with article
            len(acronym) >= 2):  # Valid acronym

            patterns.append({
                'full_text': match.group(0),
                'full_name': full_name,
                'acronym': acronym,
                'start': start,
                'end': end,
                'confidence': 0.9
            })
            print(f"  ACCEPTED as organization pattern")
        else:
            print(f"  ❌ REJECTED - doesn't meet criteria")
        print()

    return patterns

if __name__ == "__main__":
    # Test with your exact text
    test_text = "By obtaining accurate and comprehensive data on their environmental impact, they can identify areas for improvement, implement targeted strategies, and track their progress over time. Voluntary sustainability reporting frameworks like the Global Reporting Initiative (GRI), particularly GRI 13, the sector standard for agriculture, aquaculture and fishing, which came into effect in 2024, are pushing agricultural firms to allocate budget toward supply chain transparency, emissions tracking, food security, human rights and biodiversity reporting"

    print("="*80)
    print("TESTING GRI PATTERN DETECTION")
    print("="*80)

    patterns = test_organization_patterns(test_text)

    print(f"Found {len(patterns)} organization patterns:")
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern['full_text']}")
        print(f"   Full name: {pattern['full_name']}")
        print(f"   Acronym: {pattern['acronym']}")
        print()

    # Test other patterns too
    other_tests = [
        "Microsoft Corporation (MSFT) announced",
        "World Health Organization (WHO) reports",
        "Environmental Protection Agency (EPA) guidelines",
        "United Nations (UN) summit",
        "International Monetary Fund (IMF) data"
    ]

    print("="*80)
    print("TESTING OTHER ORGANIZATION PATTERNS")
    print("="*80)

    for test in other_tests:
        print(f"\nTesting: {test}")
        patterns = test_organization_patterns(test)
        if patterns:
            print(f"  Found: {patterns[0]['full_text']}")
        else:
            print(f"  No patterns found")