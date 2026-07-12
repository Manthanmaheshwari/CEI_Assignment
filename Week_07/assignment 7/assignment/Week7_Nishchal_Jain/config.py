"""
config.py
---------
Centralized configuration manager to adjust all RAG pipeline settings in one place.
This avoids hardcoding constants across different files.
"""

import os
from dotenv import load_dotenv

# Load key-value configuration values from local environment variables
load_dotenv()

# Text segmenting configuration
# SEGMENT_LIMIT specifies the target length for text chunks in characters.
# SEGMENT_STRIDE defines the overlap size to preserve context across chunks.
SEGMENT_LIMIT = 500
SEGMENT_STRIDE = 80

# Embedding model config
# Using a lightweight sentence-transformer model for CPU-efficient vector calculations
ENCODER_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

# Vector database setup
# Using inner product as metric since we normalize all embeddings to unit scale
DISTANCE_METRIC = "inner_product"
RETRIEVAL_LIMIT = 5

# Hybrid retrieval configuration
# Blends traditional lexical matching (BM25) with vector similarity.
# BLENDING_FACTOR determines the weighting (e.g. 0.7 vector / 0.3 lexical).
ENABLE_HYBRID_RETRIEVAL = True
BLENDING_FACTOR = 0.7

# Re-ranking settings
# Cross-encoder re-ranking is more precise but introduces latency.
ENABLE_CE_RERANK = False

# API configuration for generative model
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GENERATOR_MODEL = "gemini-flash-latest"

# Settings for HuggingFace datasets
REMOTE_DATA_SOURCE = "vectara/open_ragbench"
MAX_REMOTE_DOCS = 50

# Filesystem paths
LOCAL_DOCS_PATH = os.path.join(os.path.dirname(__file__), "sample_docs")
VEC_INDEX_FILE = os.path.join(os.path.dirname(__file__), "faiss_store", "doc_index.faiss")
LOGS_FOLDER = os.path.join(os.path.dirname(__file__), "logs")
LOG_OUTPUT_PATH = os.path.join(LOGS_FOLDER, "pipeline_run.log")

# Ensure required directories are created beforehand
os.makedirs(os.path.dirname(VEC_INDEX_FILE), exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
