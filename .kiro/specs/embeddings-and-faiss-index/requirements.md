# Requirements Document

## Introduction

The Med RAG system needs to build embeddings and a FAISS index from processed text chunks to enable semantic search and retrieval of relevant medical paper content. This feature will create vector representations of text chunks and build a searchable index for efficient similarity-based retrieval.

## Glossary

- **Embedding_Generator**: Component that converts text chunks into vector representations
- **FAISS_Index**: Facebook AI Similarity Search index for efficient vector similarity search
- **Chunk_Processor**: Component that processes text chunks from the ingestion pipeline
- **Vector_Store**: Combined storage of embeddings and metadata for retrieval
- **Similarity_Search**: Process of finding semantically similar chunks using vector distance

## Requirements

### Requirement 1: Generate Text Embeddings

**User Story:** As a system component, I want to generate embeddings for text chunks, so that I can perform semantic similarity search on medical paper content.

#### Acceptance Criteria

1. WHEN the system processes text chunks from the chunks.parquet file, THE Embedding_Generator SHALL create vector embeddings for each chunk
2. WHEN generating embeddings, THE Embedding_Generator SHALL use the configured embedding model from settings
3. WHEN processing chunks, THE Embedding_Generator SHALL preserve chunk metadata including doc_id, title, and page ranges
4. WHEN embedding generation fails for a chunk, THE Embedding_Generator SHALL log the error and continue processing remaining chunks
5. WHEN all chunks are processed, THE Embedding_Generator SHALL save embeddings to the configured output path

### Requirement 2: Build FAISS Index

**User Story:** As a retrieval system, I want to build a FAISS index from embeddings, so that I can perform fast similarity search across all document chunks.

#### Acceptance Criteria

1. WHEN embeddings are generated, THE FAISS_Index SHALL create an efficient search index from the embedding vectors
2. WHEN building the index, THE FAISS_Index SHALL use appropriate FAISS index type for the embedding dimensions
3. WHEN the index is built, THE FAISS_Index SHALL save the index to the configured FAISS index path
4. WHEN saving the index, THE FAISS_Index SHALL also save chunk metadata to enable result mapping
5. WHEN the index build completes, THE FAISS_Index SHALL validate the index can perform similarity searches

### Requirement 3: Handle Input Validation

**User Story:** As a system administrator, I want proper input validation, so that the embedding and indexing process fails gracefully with clear error messages.

#### Acceptance Criteria

1. WHEN the chunks.parquet file is missing, THE Chunk_Processor SHALL raise a clear error message and exit
2. WHEN the chunks.parquet file is empty or corrupted, THE Chunk_Processor SHALL raise a descriptive error and exit
3. WHEN output directories don't exist, THE Vector_Store SHALL create them automatically
4. WHEN the embedding model fails to load, THE Embedding_Generator SHALL raise a clear error with model information
5. WHEN insufficient disk space exists, THE Vector_Store SHALL detect and report the issue before processing

### Requirement 4: Progress Tracking and Logging

**User Story:** As a system operator, I want to track progress during embedding and indexing, so that I can monitor long-running operations and debug issues.

#### Acceptance Criteria

1. WHEN processing chunks, THE Embedding_Generator SHALL display progress using tqdm progress bars
2. WHEN generating embeddings, THE Embedding_Generator SHALL log batch processing statistics
3. WHEN building the FAISS index, THE FAISS_Index SHALL log index construction details including dimensions and size
4. WHEN operations complete, THE Vector_Store SHALL log summary statistics including total chunks processed and index size
5. WHEN errors occur, THE system SHALL log detailed error information with context

### Requirement 5: Batch Processing Efficiency

**User Story:** As a system component, I want efficient batch processing of embeddings, so that large document collections can be processed in reasonable time.

#### Acceptance Criteria

1. WHEN processing multiple chunks, THE Embedding_Generator SHALL process them in configurable batch sizes
2. WHEN generating embeddings, THE Embedding_Generator SHALL utilize available GPU resources if present
3. WHEN memory usage is high, THE Embedding_Generator SHALL process chunks in smaller batches to prevent out-of-memory errors
4. WHEN processing completes, THE Embedding_Generator SHALL report processing speed and total time
5. WHEN resuming interrupted processing, THE Embedding_Generator SHALL skip already processed chunks if output exists

### Requirement 6: Integration with Existing Pipeline

**User Story:** As a pipeline component, I want to integrate seamlessly with the existing ingestion pipeline, so that embeddings and indexing can be part of the automated data processing workflow.

#### Acceptance Criteria

1. WHEN called from the run_local.sh script, THE embedding and indexing process SHALL execute after chunk generation
2. WHEN chunks.parquet is updated, THE Vector_Store SHALL detect changes and rebuild embeddings if needed
3. WHEN configuration settings change, THE Embedding_Generator SHALL use the updated embedding model and paths
4. WHEN the process completes successfully, THE Vector_Store SHALL create marker files indicating successful completion
5. WHEN integration with the pipeline, THE system SHALL follow the same error handling patterns as other pipeline components