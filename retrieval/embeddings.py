"""
Embedding generation module for the Med RAG system.

This module handles the generation of vector embeddings from text chunks using
sentence-transformers. It processes chunks from the ingestion pipeline and creates
384-dimensional embeddings using the BAAI/bge-small-en-v1.5 model.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sentence-transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")


class EmbeddingGenerator:
    """
    Generates vector embeddings from text chunks using sentence-transformers.
    
    This class handles model loading, batch processing, and progress tracking
    for converting text chunks into 384-dimensional vector embeddings.
    """
    
    def __init__(self, model_name: str = None, batch_size: int = 32, device: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Number of chunks to process in each batch
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name or settings.EMBED_MODEL
        self.batch_size = batch_size
        self.device = device or self._detect_device()
        self.model: Optional[SentenceTransformer] = None
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model_name}")
        logger.info(f"Using device: {self.device}, batch size: {self.batch_size}")
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device (GPU/CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Using CPU for embedding generation")
        return device
    
    def load_model(self) -> None:
        """
        Load the sentence-transformers model.
        
        Raises:
            SystemExit: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Verify model dimensions
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            embedding_dim = test_embedding.shape[1]
            logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
            
            if embedding_dim != 384:
                logger.warning(f"Expected 384 dimensions, got {embedding_dim}")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise SystemExit(f"Model loading failed: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            ValueError: If model is not loaded or texts is empty
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return np.array([]).reshape(0, 384)
        
        try:
            # Generate embeddings with progress bar for large batches
            show_progress = len(texts) > 100
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def process_chunks_file(self, input_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Process chunks from a parquet file and generate embeddings.
        
        Args:
            input_path: Path to the chunks.parquet file
            
        Returns:
            Tuple of (embeddings_array, metadata_dataframe)
            
        Raises:
            SystemExit: If input file is missing or corrupted
        """
        # Validate input file
        if not input_path.exists():
            raise SystemExit(f"Missing input file: {input_path}")
        
        try:
            logger.info(f"Loading chunks from: {input_path}")
            df = pd.read_parquet(input_path)
            
            if df.empty:
                raise SystemExit(f"Empty chunks file: {input_path}")
            
            # Validate required columns
            required_columns = ["chunk_id", "doc_id", "title", "page_start", "page_end", "chunk_text"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise SystemExit(f"Missing required columns in {input_path}: {missing_columns}")
            
            logger.info(f"Loaded {len(df)} chunks from {len(df['doc_id'].unique())} documents")
            
        except Exception as e:
            logger.error(f"Failed to load chunks file: {e}")
            raise SystemExit(f"Corrupted chunks file: {e}")
        
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Extract texts for embedding
        texts = df["chunk_text"].tolist()
        
        # Process in batches with progress tracking
        logger.info("Generating embeddings...")
        all_embeddings = []
        failed_chunks = []
        
        # Process chunks in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_indices = list(range(i, min(i + self.batch_size, len(texts))))
            
            try:
                batch_embeddings = self.generate_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
                
                # Log batch statistics
                if i % (self.batch_size * 10) == 0:  # Log every 10 batches
                    logger.debug(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} chunks")
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                # Track failed chunks but continue processing
                failed_chunks.extend(batch_indices)
                # Create zero embeddings for failed chunks to maintain alignment
                zero_embeddings = np.zeros((len(batch_texts), 384))
                all_embeddings.append(zero_embeddings)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([]).reshape(0, 384)
        
        # Log processing statistics
        total_chunks = len(texts)
        successful_chunks = total_chunks - len(failed_chunks)
        logger.info(f"Embedding generation complete:")
        logger.info(f"  Total chunks: {total_chunks}")
        logger.info(f"  Successful: {successful_chunks}")
        logger.info(f"  Failed: {len(failed_chunks)}")
        logger.info(f"  Embedding shape: {embeddings.shape}")
        
        if failed_chunks:
            logger.warning(f"Failed to generate embeddings for {len(failed_chunks)} chunks")
            # Log first few failed chunk IDs for debugging
            failed_chunk_ids = [df.iloc[idx]["chunk_id"] for idx in failed_chunks[:5]]
            logger.warning(f"First few failed chunks: {failed_chunk_ids}")
        
        return embeddings, df


def main():
    """
    Main function to generate embeddings from chunks.parquet file.
    
    This function serves as the entry point for the embedding generation process.
    It loads chunks, generates embeddings, and saves the results.
    """
    # Set up paths
    input_path = Path(settings.CHUNKS_PARQUET)
    output_dir = Path(settings.PROCESSED_DIR)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Process chunks and generate embeddings
    embeddings, metadata = generator.process_chunks_file(input_path)
    
    # Save embeddings (will be used by FAISS index builder)
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to: {embeddings_path}")
    
    # Save metadata with embedding indices
    metadata_with_indices = metadata.copy()
    metadata_with_indices["embedding_index"] = range(len(metadata))
    
    metadata_path = output_dir / "embeddings_meta.parquet"
    metadata_with_indices.to_parquet(metadata_path, index=False)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    logger.info("✅ Embedding generation completed successfully")


if __name__ == "__main__":
    main()