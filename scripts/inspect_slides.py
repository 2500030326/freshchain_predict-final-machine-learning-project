import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

for p in sorted(Path('extracted_slides').glob('*.json')):
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    n = len(data['slides'])
    print(f"\n==========================================")
    print(f"DECK: {p.name} ({n} slides)")
    print(f"==========================================")
    for s in data['slides']:
        elements = [t.replace('\n', ' ').strip() for t in s['text_elements'] if t.strip()]
        title = elements[0] if elements else "NO TITLE"
        snippet = " | ".join(elements[1:4]) if len(elements) > 1 else ""
        print(f"Slide {s['slide_number']:2d}: {title[:45]} --> {snippet[:60]}")
