import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import re

PPTX_DIR = Path(r"C:\Users\tejas\Downloads\Telegram Desktop")
TARGET_FILES = [
    "Probability_Calibration_Placement.pptx",
    "module_5_session_M5_S1 (L59-60).pptx",
    "module_5_session_M5_S2 (L61-62).pptx",
    "Module_5_Session_M5-3.pptx",
    "Module_5_Session_M5-4.pptx",
    "Module_5_Session_M5-5.pptx",
    "Module_5_Session_M5-6_to_M5-8.pptx",
    "Cluster_Augmented_Regression (1).pptx",
    "M4-Session1-KMeans_clustering-new.pptx",
    "M4-Session2-Hierarchical_Clustering-new.pptx",
]

OUT_DIR = Path("extracted_slides")
OUT_DIR.mkdir(exist_ok=True)

def extract_text_from_xml(xml_content):
    root = ET.fromstring(xml_content)
    texts = []
    # Namespaces commonly used in DrawingML / PresentationML
    # Find all text elements <a:t>
    for elem in root.iter():
        if elem.tag.endswith('}t'):
            if elem.text:
                texts.append(elem.text)
    return texts

def extract_deck(file_name):
    pptx_path = PPTX_DIR / file_name
    if not pptx_path.exists():
        print(f"File not found: {pptx_path}")
        return None
    
    deck_data = {"file_name": file_name, "slides": []}
    with zipfile.ZipFile(pptx_path, "r") as z:
        # Sort slides by slide number
        slide_names = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
        
        def slide_num(s):
            m = re.search(r'slide(\d+)\.xml', s)
            return int(m.group(1)) if m else 999
            
        slide_names.sort(key=slide_num)
        
        for idx, s_name in enumerate(slide_names, 1):
            xml_content = z.read(s_name)
            texts = extract_text_from_xml(xml_content)
            deck_data["slides"].append({
                "slide_number": idx,
                "xml_name": s_name,
                "text_elements": texts,
                "full_text": "\n".join(texts)
            })
            
    out_file = OUT_DIR / (Path(file_name).stem + ".json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(deck_data, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(deck_data['slides'])} slides from {file_name} -> {out_file}")
    return deck_data

def main():
    summary = {}
    for fname in TARGET_FILES:
        res = extract_deck(fname)
        if res:
            summary[fname] = len(res["slides"])
    print("\nExtraction Summary:")
    for k, v in summary.items():
        print(f" - {k}: {v} slides")

if __name__ == "__main__":
    main()
