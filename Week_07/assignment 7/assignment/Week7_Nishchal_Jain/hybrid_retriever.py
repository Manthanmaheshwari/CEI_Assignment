"""
hybrid_retriever.py
-------------------
Implements BM25 lexical search, score-fused hybrid retrieval (lexical + vector),
and a second-stage cross-encoder re-ranking step.
"""

import math
import logging
from collections import Counter

app_logger = logging.getLogger(__name__)


# ======================================================================
#  Lexical Search Engine (BM25 Implementation)
# ======================================================================

class KeywordSearchEngine:
    """
    Custom BM25 retrieval module written from scratch to score segments.
    Applies standard term frequency saturation and document length normalization.
    """

    def __init__(self, segments, k1=1.5, b=0.75):
        """
        Builds the lexical index.
        k1: saturation threshold (higher = less sensitive to repetition)
        b: length penalty weight (0 = no penalty, 1 = full penalty)
        """
        self.k1 = k1
        self.b = b
        self.segment_registry = segments
        self.total_docs_count = len(segments)

        self.tokenized_docs = []
        self.doc_token_lengths = []
        self.term_document_frequencies = Counter()

        for segment in segments:
            word_tokens = self._lex_text(segment["chunk_text"])
            self.tokenized_docs.append(word_tokens)
            self.doc_token_lengths.append(len(word_tokens))

            # Record occurrences of terms for document frequency calculation
            distinct_words = set(word_tokens)
            for word in distinct_words:
                self.term_document_frequencies[word] += 1

        self.average_length = sum(self.doc_token_lengths) / max(self.total_docs_count, 1)
        app_logger.info(f"Lexical index compiled. Document count: {self.total_docs_count}, Unique vocabulary: {len(self.term_document_frequencies)}")

    def _lex_text(self, raw_string):
        """
        Converts text to lowercase and tokenizes using simple regular expression.
        """
        import re
        lexed_tokens = re.findall(r'[a-z0-9]+', raw_string.lower())
        return lexed_tokens

    def _get_idf_value(self, word):
        """
        Computes the inverse document frequency. Adds 0.5 smoothing.
        """
        doc_count_with_word = self.term_document_frequencies.get(word, 0)
        return math.log((self.total_docs_count - doc_count_with_word + 0.5) / (doc_count_with_word + 0.5) + 1.0)

    def retrieve(self, search_query, results_count=5):
        """
        Scores all segments against the user search query.
        Returns top matching segments.
        """
        query_words = self._lex_text(search_query)

        scored_results = []
        for doc_position in range(self.total_docs_count):
            accumulated_score = 0.0
            current_length = self.doc_token_lengths[doc_position]
            token_counter = Counter(self.tokenized_docs[doc_position])

            for word in query_words:
                if word not in token_counter:
                    continue

                frequency = token_counter[word]
                idf_weight = self._get_idf_value(word)

                numerator_val = frequency * (self.k1 + 1)
                denominator_val = frequency + self.k1 * (1 - self.b + self.b * (current_length / self.average_length))
                accumulated_score += idf_weight * (numerator_val / denominator_val)

            scored_results.append((doc_position, accumulated_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        highest_scoring_indices = scored_results[:results_count]

        matching_segments = []
        for match_pos, accumulated_score in highest_scoring_indices:
            if accumulated_score > 0:
                copied_chunk = self.segment_registry[match_pos].copy()
                copied_chunk["bm25_score"] = accumulated_score
                matching_segments.append(copied_chunk)

        return matching_segments


# ======================================================================
#  Hybrid Search Fusion (Vector Similarity + BM25 Score Fusion)
# ======================================================================

def combined_hybrid_retrieve(faiss_db, lexical_engine, search_vector, transformer_model,
                             search_query, segment_list, num_results=5, weight_factor=0.7):
    """
    Fuses findings from FAISS semantic indexing and BM25 lexical querying.
    Scores from each retriever are scaled to the [0, 1] range and linearly combined.
    weight_factor: semantic weight ratio (alpha).
    """
    from vector_store import retrieve_nearest_neighbors

    # Fetch larger candidate pool from both algorithms for score normalization
    retrieval_pool_size = min(num_results * 3, len(segment_list))

    # Vector lookup
    semantic_hits = retrieve_nearest_neighbors(faiss_db, search_vector, segment_list, num_results=retrieval_pool_size)

    # Keyword lookup
    lexical_hits = lexical_engine.retrieve(search_query, results_count=retrieval_pool_size)

    # Merge candidates into map for min-max scaling
    blended_scores = {}

    for candidate_chunk in semantic_hits:
        chunk_num = candidate_chunk["chunk_index"]
        origin = candidate_chunk["source"]
        lookup_key = f"{origin}_{chunk_num}"
        if lookup_key not in blended_scores:
            blended_scores[lookup_key] = {"vector": 0.0, "keyword": 0.0, "chunk": candidate_chunk}
        blended_scores[lookup_key]["vector"] = candidate_chunk.get("relevance_score", 0.0)

    for candidate_chunk in lexical_hits:
        chunk_num = candidate_chunk["chunk_index"]
        origin = candidate_chunk["source"]
        lookup_key = f"{origin}_{chunk_num}"
        if lookup_key not in blended_scores:
            blended_scores[lookup_key] = {"vector": 0.0, "keyword": 0.0, "chunk": candidate_chunk}
        blended_scores[lookup_key]["keyword"] = candidate_chunk.get("bm25_score", 0.0)

    # Normalize vectors and BM25 score outputs to standard range
    semantic_scores_pool = [v["vector"] for v in blended_scores.values()]
    lexical_scores_pool = [v["keyword"] for v in blended_scores.values()]

    min_semantic = min(semantic_scores_pool) if semantic_scores_pool else 0
    max_semantic = max(semantic_scores_pool) if semantic_scores_pool else 1
    min_lexical = min(lexical_scores_pool) if lexical_scores_pool else 0
    max_lexical = max(lexical_scores_pool) if lexical_scores_pool else 1

    semantic_spread = max_semantic - min_semantic if max_semantic != min_semantic else 1.0
    lexical_spread = max_lexical - min_lexical if max_lexical != min_lexical else 1.0

    # Calculate blended scores
    final_scores_list = []
    for lookup_key, payload in blended_scores.items():
        scaled_semantic = (payload["vector"] - min_semantic) / semantic_spread
        scaled_lexical = (payload["keyword"] - min_lexical) / lexical_spread

        fused_score = weight_factor * scaled_semantic + (1 - weight_factor) * scaled_lexical

        copied_candidate = payload["chunk"].copy()
        copied_candidate["hybrid_score"] = fused_score
        copied_candidate["vector_score_norm"] = scaled_semantic
        copied_candidate["keyword_score_norm"] = scaled_lexical
        copied_candidate["relevance_score"] = fused_score
        final_scores_list.append(copied_candidate)

    final_scores_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

    app_logger.debug(
        f"Hybrid retrieval finished: blended {len(semantic_hits)} semantic and {len(lexical_hits)} lexical "
        f"candidates to return {min(num_results, len(final_scores_list))} top matches"
    )

    return final_scores_list[:num_results]


# ======================================================================
#  Second-Stage Cross-Encoder Re-Ranking
# ======================================================================

_global_reranker = None


def rerank_candidates(search_query, retrieved_candidates, num_results=5):
    """
    Reranks the retrieved context segments using a pre-trained CrossEncoder.
    Processes query and segment together to capture deep semantic relevance.
    """
    global _global_reranker

    if not retrieved_candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        app_logger.warning("CrossEncoder not found. Skipping re-ranking phase")
        return retrieved_candidates[:num_results]

    # Initialize model if it has not been loaded in the process
    if _global_reranker is None:
        app_logger.info("Initializing re-ranking CrossEncoder model...")
        _global_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        app_logger.info("Re-ranking model loaded successfully")

    query_chunk_pairs = [(search_query, candidate["chunk_text"]) for candidate in retrieved_candidates]

    prediction_scores = _global_reranker.predict(query_chunk_pairs)

    for pos, candidate in enumerate(retrieved_candidates):
        candidate["rerank_score"] = float(prediction_scores[pos])
        # Update standard key for downstream compatibility
        candidate["relevance_score"] = float(prediction_scores[pos])

    reordered_candidates = sorted(retrieved_candidates, key=lambda x: x["rerank_score"], reverse=True)

    app_logger.debug(
        f"Re-ranking complete: top candidate score modified from {retrieved_candidates[0].get('hybrid_score', 0):.4f} "
        f"to {reordered_candidates[0]['rerank_score']:.4f}"
    )

    return reordered_candidates[:num_results]
