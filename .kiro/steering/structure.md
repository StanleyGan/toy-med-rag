# Project Structure

## Directory Organization

```
├── api/                    # FastAPI web service endpoints
├── config/                 # Configuration and settings
│   └── settings.py        # Centralized Pydantic settings
├── data/                  # Data storage (gitignored)
│   ├── raw_pdfs/         # Input PDF files
│   └── processed/        # Generated data files
├── evaluation/           # Model evaluation and metrics
├── generation/           # LLM prompts and response generation
│   └── prompts.py       # System and user prompt templates
├── ingestion/           # PDF processing pipeline
│   ├── parse_pdfs.py   # PDF text extraction
│   └── chunk_text.py   # Text chunking with overlap
├── retrieval/          # Vector search and document retrieval
└── scripts/            # Automation and utility scripts
    └── run_local.sh   # Full ingestion pipeline
```

## Key Files

### Configuration
- `config/settings.py`: Single source of truth for all configuration
- Uses Pydantic for validation and type safety
- Environment variables via `.env` file

### Data Pipeline
- `ingestion/parse_pdfs.py`: Converts PDFs to JSONL format
- `ingestion/chunk_text.py`: Creates overlapping text chunks
- Output: `data/processed/docs.jsonl` and `data/processed/chunks.parquet`

### Generation
- `generation/prompts.py`: Contains all prompt templates
- Includes system prompt, answer template, and verifier template
- Structured output format enforcement

## Naming Conventions

### Files
- Snake_case for Python files and directories
- Descriptive names indicating purpose (e.g., `parse_pdfs.py`, `chunk_text.py`)

### Data
- Document IDs: Use PDF filename stem as `doc_id`
- Chunk IDs: Format `{doc_id}::c{index:04d}`
- Citations: Include both document ID and page ranges

### Functions
- Descriptive function names with clear purpose
- Private functions prefixed with underscore (e.g., `_sliding_window`)

## Architecture Patterns

### Settings Management
- Centralized configuration in `config/settings.py`
- Import settings object: `from config.settings import settings`
- Environment-specific overrides via `.env`

### Data Flow
1. Raw PDFs → `parse_pdfs.py` → JSONL documents
2. JSONL → `chunk_text.py` → Parquet chunks with page attribution
3. Chunks → embeddings → FAISS index (future)
4. Query → retrieval → generation → structured response

### Error Handling
- Use `SystemExit` for pipeline failures with descriptive messages
- Validate input data existence before processing
- Create output directories automatically with `mkdir(parents=True, exist_ok=True)`