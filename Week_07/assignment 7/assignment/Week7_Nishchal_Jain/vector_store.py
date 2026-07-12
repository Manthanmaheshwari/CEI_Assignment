"""
vector_store.py
---------------
Manages the generation, persistence, loading, and querying of FAISS vector indices.
Allows fast nearest-neighbor retrieval of semantically similar segments.
"""

import os
import numpy as np
import faiss
import logging

app_logger = logging.getLogger(__name__)


def create_faiss_store(vector_data):
    """
    Initializes a flat inner-product FAISS index using normalized vector data.
    Inner product functions as cosine similarity for L2-normalized vectors.
    """
    if vector_data.size == 0:
        app_logger.error("Unable to initialize vector store from empty array")
        return None

    if not isinstance(vector_data, np.ndarray):
        vector_data = np.array(vector_data, dtype=np.float32)
    if vector_data.dtype != np.float32 or not vector_data.flags["C_CONTIGUOUS"]:
        vector_data = np.ascontiguousarray(vector_data, dtype=np.float32)

    feature_dims = vector_data.shape[1]
    vector_count = vector_data.shape[0]

    app_logger.info(f"Generating FAISS flat index: storing {vector_count} items with {feature_dims} dimensions")

    # IndexFlatIP handles inner product similarity
    faiss_index_obj = faiss.IndexFlatIP(feature_dims)
    faiss_index_obj.add(vector_data)

    app_logger.info(f"FAISS index construction complete. Stored vectors count: {faiss_index_obj.ntotal}")
    return faiss_index_obj


def retrieve_nearest_neighbors(faiss_index_obj, search_vector, segment_list, num_results=5):
    """
    Queries the FAISS index to find the most relevant document segments.
    Maps matches to original segments and appends scores.
    """
    if faiss_index_obj is None or faiss_index_obj.ntotal == 0:
        app_logger.warning("Empty vector index. Aborting query search")
        return []

    if not isinstance(search_vector, np.ndarray):
        search_vector = np.array(search_vector, dtype=np.float32)
    if len(search_vector.shape) == 1:
        search_vector = search_vector.reshape(1, -1)
    if search_vector.dtype != np.float32 or not search_vector.flags["C_CONTIGUOUS"]:
        search_vector = np.ascontiguousarray(search_vector, dtype=np.float32)

    if search_vector.shape[1] != faiss_index_obj.d:
        app_logger.error(
            f"Query dimension mismatch: vector has {search_vector.shape[1]} features, "
            f"index expects {faiss_index_obj.d}"
        )
        return []

    # Bind the retrieval count to the number of indexed vectors
    bounded_k = min(num_results, faiss_index_obj.ntotal)

    # Perform nearest neighbor search
    matching_scores, candidate_indices = faiss_index_obj.search(search_vector, bounded_k)

    nearest_matches = []
    for position, (metric_val, vector_idx) in enumerate(zip(matching_scores[0], candidate_indices[0])):
        if vector_idx == -1 or vector_idx < 0 or vector_idx >= len(segment_list):
            continue

        copied_segment = segment_list[vector_idx].copy()
        copied_segment["relevance_score"] = float(metric_val)
        copied_segment["retrieval_rank"] = position + 1
        nearest_matches.append(copied_segment)

    if nearest_matches:
        app_logger.debug(f"Retrieved {len(nearest_matches)} chunks (highest similarity: {nearest_matches[0]['relevance_score']:.4f})")
    else:
        app_logger.debug("No semantic matches found in vector index")

    return nearest_matches


def persist_vector_index(faiss_index_obj, output_path):
    """
    Saves the constructed FAISS index to the local filesystem for future runs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(faiss_index_obj, output_path)
    app_logger.info(f"FAISS index successfully saved to disk: {output_path}")


def restore_vector_index(input_path):
    """
    Loads an existing FAISS index file from local storage.
    """
    if not os.path.exists(input_path):
        app_logger.warning(f"No FAISS database file located at {input_path}")
        return None

    loaded_index = faiss.read_index(input_path)
    app_logger.info(f"Restored FAISS index from file: {input_path} (contains {loaded_index.ntotal} records)")
    return loaded_index
