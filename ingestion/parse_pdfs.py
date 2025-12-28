import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import fitz  # PyMuPDF
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append({
            "page_number": i + 1,  # human-friendly
            "text": text.strip()
        })
    doc.close()
    return pages


def main():
    raw_dir = Path(settings.RAW_PDF_DIR)
    out_path = Path(settings.DOCS_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in raw_dir.glob("*.pdf")])
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {raw_dir}. Put files into data/raw_pdfs/")

    with out_path.open("w", encoding="utf-8") as f_out:
        for pdf in tqdm(pdf_files, desc="Parsing PDFs"):
            doc_id = pdf.stem
            pages = extract_pdf_pages(pdf)

            # Very simple metadata v0 (improve later)
            record = {
                "doc_id": doc_id,
                "source_file": pdf.name,
                "title": doc_id,     # placeholder; you can enrich later
                "year": None,
                "pages": pages,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
