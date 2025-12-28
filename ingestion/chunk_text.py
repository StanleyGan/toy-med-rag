import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def _sliding_window(text: str, max_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, chunk_text) using char-based heuristic."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks


def main():
    docs_path = Path(settings.DOCS_JSONL)
    out_path = Path(settings.CHUNKS_PARQUET)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not docs_path.exists():
        raise SystemExit(f"Missing {docs_path}. Run ingestion/parse_pdfs.py first.")

    all_rows: List[Dict[str, Any]] = []

    with docs_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Chunking docs"):
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            title = doc.get("title") or doc_id
            pages = doc.get("pages", [])

            # Concatenate pages but keep a page map so we can attribute chunk spans back to pages
            # We'll build a single full_text with markers per page.
            page_offsets = []  # (page_number, start_char, end_char)
            full_parts = []
            cursor = 0
            for p in pages:
                t = (p.get("text") or "").strip()
                if not t:
                    continue
                start = cursor
                full_parts.append(t + "\n")
                cursor += len(t) + 1
                end = cursor
                page_offsets.append((p["page_number"], start, end))

            full_text = "".join(full_parts).strip()
            if not full_text:
                continue

            windows = _sliding_window(
                full_text,
                max_chars=settings.CHUNK_MAX_CHARS,
                overlap_chars=settings.CHUNK_OVERLAP_CHARS,
            )

            for idx, (s, e, chunk_text) in enumerate(windows):
                # Map char span to page range
                pages_hit = [pn for (pn, ps, pe) in page_offsets if not (e <= ps or s >= pe)]
                if pages_hit:
                    page_start, page_end = min(pages_hit), max(pages_hit)
                else:
                    page_start, page_end = None, None

                all_rows.append({
                    "chunk_id": f"{doc_id}::c{idx:04d}",
                    "doc_id": doc_id,
                    "title": title,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_text": chunk_text,
                })

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    print(f"✅ Wrote: {out_path}  ({len(df)} chunks)")


if __name__ == "__main__":
    main()
