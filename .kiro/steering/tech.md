
# Technology Stack

## Core Framework
- **FastAPI**: Web API framework for serving endpoints
- **Pydantic**: Data validation and settings management
- **Python 3.x**: Primary language

## PDF Processing
- **PyMuPDF (fitz)**: PDF parsing and text extraction
- Character-based chunking with sliding window approach

## Data & Storage
- **Pandas**: Data manipulation and processing
- **PyArrow/Parquet**: Efficient data storage format
- **JSONL**: Document storage format

## Vector Search & Embeddings
- **sentence-transformers**: Text embeddings (default: BAAI/bge-small-en-v1.5)
- **FAISS**: Vector similarity search and indexing
- **NumPy**: Numerical operations

## Development Tools
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bars for long-running operations
- **uvicorn**: ASGI server for FastAPI

## Common Commands

### Initial Setup
```bash
pip install -r requirements.txt
```

### Data Processing Pipeline
```bash
# Full ingestion pipeline
./scripts/run_local.sh

# Individual steps
python ingestion/parse_pdfs.py    # Extract text from PDFs
python ingestion/chunk_text.py    # Create text chunks
```

### Development
```bash
# Start API server (when implemented)
uvicorn api.main:app --reload --port 8000
```

## Configuration
- Settings managed via `config/settings.py` using Pydantic
- Environment variables loaded from `.env` file
- Key paths and parameters configurable via Settings class