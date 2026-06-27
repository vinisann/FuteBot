import sys
import os

# Adjust path to find src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force stdout to be utf-8
sys.stdout.reconfigure(encoding='utf-8')

from src.scraper import fetch_match_specific_news

print("=== Running match-specific news scrape test (Brasil vs Escócia) ===")
results = fetch_match_specific_news("Brasil", "Escócia", max_results=3)

print(f"Found {len(results)} news items.")
for i, item in enumerate(results):
    print(f"\n[{i+1}] Title: {item['title']}")
    print(f"    Source: {item['source']}")
    print(f"    Link: {item['link']}")
    print(f"    Pub Date: {item['pub_date']}")
    if item['parsed_lineup']:
        print("    --> Lineup Parsed:")
        # Indent lineup info
        indented = "        " + item['parsed_lineup'].replace("\n", "\n        ")
        print(indented)
    else:
        print("    --> (No probable lineup parsed from this page)")

print("\n=== Test complete! ===")
