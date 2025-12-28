# Product Overview

Med RAG Flagship is an evidence-grounded medical paper Q&A system that answers questions using ONLY provided medical research papers.

## Core Functionality
- Processes PDF medical research papers through ingestion pipeline
- Provides concise answers with evidence-based citations
- Includes page-range citations for traceability
- Explicitly states limitations and uncertainty
- Refuses to answer when evidence is insufficient

## Key Principles
- **Evidence-only responses**: No external knowledge beyond provided papers
- **Citation requirements**: Every claim must be supported with document + page citations
- **Uncertainty transparency**: Explicitly communicate limitations and gaps
- **Refusal over speculation**: Better to refuse than provide unsupported answers

## Output Format
All responses follow a structured format:
- Concise answer (≤120 words)
- Evidence bullets with citations (doc + page range)
- Limitations/uncertainty section
- Refusal template when evidence is insufficient