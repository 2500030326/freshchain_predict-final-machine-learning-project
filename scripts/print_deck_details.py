import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

deck_name = sys.argv[1] if len(sys.argv) > 1 else "Cluster_Augmented_Regression (1).json"
p = Path("extracted_slides") / deck_name

with open(p, encoding="utf-8") as f:
    d = json.load(f)

print(f"Total Slides: {len(d['slides'])}")
for s in d['slides']:
    print(f"\n==========================================")
    print(f"SLIDE {s['slide_number']}")
    print(f"==========================================")
    print(s['full_text'].strip())
