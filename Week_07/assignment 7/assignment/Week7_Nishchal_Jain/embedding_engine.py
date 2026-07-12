"""
embedding_engine.py
-------------------
Wraps the sentence-transformers library to manage text-to-vector operations.
Encodes text data into dense vectors representing semantic meanings.
"""

import numpy as np
import logging

app_logger = logging.getLogger(__name__)


def initialize_vector_model(model_name="all-MiniLM-L6-v2"):
    """
    Initializes and caches the SentenceTransformer model.
    Downloads the weights on first run, loads from cache on subsequent calls.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        app_logger.error("sentence-transformers package is missing. Run: pip install sentence-transformers")
        raise

    app_logger.info(f"Initializing embedding model: {model_name}")
    transformer_model = SentenceTransformer(model_name)

    # Perform a fast sanity check on model dimensions
    dummy_vector = transformer_model.encode(["dimension verification test"])
    dim_size = dummy_vector.shape[1]
    app_logger.info(f"Model loaded and verified. Vector dimensionality: {dim_size}")

    return transformer_model


def generate_chunk_embeddings(transformer_model, segments, size_limit=64):
    """
    Processes a list of text segments and converts them to vector representations in batches.
    Normalizes embeddings to unit length to allow inner product computations for cosine similarity.
    """
    if not segments:
        app_logger.warning("Empty segments list passed to embedder")
        return np.array([])

    raw_texts = [segment["chunk_text"] for segment in segments]

    app_logger.info(f"Encoding {len(raw_texts)} text blocks using batch size: {size_limit}")

    computed_vectors = transformer_model.encode(
        raw_texts,
        batch_size=size_limit,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    vector_matrix = np.array(computed_vectors, dtype=np.float32)
    app_logger.info(f"Created embedding matrix of shape: {vector_matrix.shape}")

    return vector_matrix


def encode_search_query(transformer_model, search_phrase):
    """
    Encodes a single query string. Normalizes the output vector to enable
    comparisons using inner products.
    """
    encoded_phrase = transformer_model.encode(
        [search_phrase],
        normalize_embeddings=True
    )

    return np.array(encoded_phrase, dtype=np.float32)
