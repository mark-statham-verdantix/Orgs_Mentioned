#!/usr/bin/env python3
"""Test the regex fix for parentheses"""

import re

def test_regex_patterns():
    """Test different regex patterns"""

    test_text = "Voluntary sustainability reporting frameworks like the Global Reporting Initiative (GRI), particularly GRI 13"
    test_text_lower = test_text.lower()

    test_terms = [
        'Global Reporting Initiative (GRI)',
        'global reporting initiative (gri)',
        'GRI',
        'gri'
    ]

    for term in test_terms:
        print(f"\nTesting term: '{term}'")
        term_lower = term.lower()

        # Test old method (word boundaries)
        old_pattern = r'\b' + re.escape(term_lower) + r'\b'
        old_matches = list(re.finditer(old_pattern, test_text_lower))

        # Test new method (lookahead/lookbehind for parentheses)
        if '(' in term_lower or ')' in term_lower:
            new_pattern = r'(?<!\w)' + re.escape(term_lower) + r'(?!\w)'
        else:
            new_pattern = r'\b' + re.escape(term_lower) + r'\b'
        new_matches = list(re.finditer(new_pattern, test_text_lower))

        print(f"  Old pattern: {old_pattern}")
        print(f"  Old matches: {len(old_matches)}")
        if old_matches:
            for i, match in enumerate(old_matches):
                matched_text = test_text[match.start():match.end()]
                print(f"    Match {i+1}: '{matched_text}' at {match.start()}-{match.end()}")

        print(f"  New pattern: {new_pattern}")
        print(f"  New matches: {len(new_matches)}")
        if new_matches:
            for i, match in enumerate(new_matches):
                matched_text = test_text[match.start():match.end()]
                print(f"    Match {i+1}: '{matched_text}' at {match.start()}-{match.end()}")

        print(f"  Improvement: {len(new_matches) - len(old_matches)} additional matches")

if __name__ == "__main__":
    print("="*80)
    print("TESTING REGEX PATTERNS FOR PARENTHESES")
    print("="*80)
    test_regex_patterns()