#!/usr/bin/env bash
set -e

python ingestion/parse_pdfs.py
python ingestion/chunk_text.py

echo "✅ Ingestion complete."
echo "Next: build embeddings + FAISS index (we’ll add this next)."
