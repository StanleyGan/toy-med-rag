# Med RAG Flagship (Evidence-grounded medical paper Q&A)

This system answers questions using ONLY the provided medical research papers.
It outputs:
- concise answer
- evidence bullets with citations (doc + page range)
- limitations/uncertainty
- refusal when evidence is insufficient

## Quickstart
1) Put PDFs into `data/raw_pdfs/`
2) Run:
```bash
python ingestion/parse_pdfs.py
python ingestion/chunk_text.py
