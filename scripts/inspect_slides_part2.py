import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

decks = [
    'Module_5_Session_M5-3.json',
    'Module_5_Session_M5-4.json',
    'Module_5_Session_M5-5.json',
    'M4-Session1-KMeans_clustering-new.json',
    'M4-Session2-Hierarchical_Clustering-new.json'
]

for d in decks:
    p = Path('extracted_slides') / d
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n==========================================")
    print(f"DECK: {d} ({len(data['slides'])} slides)")
    print(f"==========================================")
    for s in data['slides']:
        elements = [t.replace('\n', ' ').strip() for t in s['text_elements'] if t.strip()]
        title = elements[0] if elements else "NO TITLE"
        snippet = " | ".join(elements[1:4]) if len(elements) > 1 else ""
        print(f"Slide {s['slide_number']:2d}: {title[:45]} --> {snippet[:60]}")
