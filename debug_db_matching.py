#!/usr/bin/env python3
"""Debug database matching for GRI"""

import re
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def test_db_lookup():
    """Test database lookup for GRI"""

    # Connect to database
    try:
        postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DATABASE', 'postgres'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }

        conn = psycopg2.connect(
            host=postgres_config['host'],
            port=postgres_config.get('port', 5432),
            database=postgres_config['database'],
            user=postgres_config['user'],
            password=postgres_config['password']
        )

        print("Connected to database")

        # Search for GRI in the database
        cursor = conn.cursor()

        # Search variations
        search_terms = [
            'Global Reporting Initiative (GRI)',
            'Global Reporting Initiative',
            'GRI',
            '%global reporting%',
            '%GRI%'
        ]

        for term in search_terms:
            print(f"\nSearching for: '{term}'")

            if '%' in term:
                cursor.execute("""
                    SELECT org_id, org_name
                    FROM verdantix.org
                    WHERE org_name ILIKE %s
                    ORDER BY org_name
                """, (term,))
            else:
                cursor.execute("""
                    SELECT org_id, org_name
                    FROM verdantix.org
                    WHERE org_name = %s
                    ORDER BY org_name
                """, (term,))

            results = cursor.fetchall()

            if results:
                print(f"  Found {len(results)} matches:")
                for org_id, org_name in results:
                    print(f"    ID: {org_id}, Name: '{org_name}'")
            else:
                print(f"  No matches found")

        conn.close()

    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def test_text_matching():
    """Test text matching logic"""

    test_text = "By obtaining accurate and comprehensive data on their environmental impact, they can identify areas for improvement, implement targeted strategies, and track their progress over time. Voluntary sustainability reporting frameworks like the Global Reporting Initiative (GRI), particularly GRI 13, the sector standard for agriculture, aquaculture and fishing, which came into effect in 2024, are pushing agricultural firms to allocate budget toward supply chain transparency, emissions tracking, food security, human rights and biodiversity reporting"

    print("\n" + "="*80)
    print("TESTING TEXT MATCHING LOGIC")
    print("="*80)

    print(f"Test text length: {len(test_text)}")
    print(f"Test text: {test_text[:100]}...")

    # Simulate the lookup logic
    text_lower = test_text.lower()

    # Test different terms
    test_terms = [
        'Global Reporting Initiative (GRI)',
        'global reporting initiative (gri)',
        'Global Reporting Initiative',
        'global reporting initiative',
        'GRI',
        'gri'
    ]

    for term in test_terms:
        print(f"\nTesting term: '{term}'")
        term_lower = term.lower()

        # Check if term appears in text
        if term_lower in text_lower:
            print(f"  Term found in text (basic substring)")

            # Test regex boundary matching
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            matches = list(re.finditer(pattern, text_lower))

            if matches:
                print(f"  Regex boundary match found ({len(matches)} matches)")
                for i, match in enumerate(matches):
                    start, end = match.span()
                    matched_text = test_text[start:end]
                    print(f"    Match {i+1}: '{matched_text}' at position {start}-{end}")
            else:
                print(f"  NO regex boundary match")
                print(f"     Pattern: {pattern}")

                # Try to find why it failed
                simple_matches = [m.start() for m in re.finditer(re.escape(term_lower), text_lower)]
                if simple_matches:
                    print(f"     Simple matches at positions: {simple_matches[:3]}")
                    # Show context around first match
                    pos = simple_matches[0]
                    start_context = max(0, pos - 10)
                    end_context = min(len(text_lower), pos + len(term_lower) + 10)
                    context = text_lower[start_context:end_context]
                    print(f"     Context: '{context}'")

                    # Check what characters are around the term
                    char_before = text_lower[pos-1] if pos > 0 else 'START'
                    char_after = text_lower[pos + len(term_lower)] if pos + len(term_lower) < len(text_lower) else 'END'
                    print(f"     Character before: '{char_before}' (is word boundary: {not char_before.isalnum()})")
                    print(f"     Character after: '{char_after}' (is word boundary: {not char_after.isalnum()})")
        else:
            print(f"  Term NOT found in text")

if __name__ == "__main__":
    print("="*80)
    print("DEBUGGING GRI DATABASE MATCHING")
    print("="*80)

    test_db_lookup()
    test_text_matching()