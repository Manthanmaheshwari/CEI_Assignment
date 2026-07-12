"""
rag_pipeline.py
---------------
Ties the document loader, text chunker, embedding engine, vector store, and 
answer generator together into a complete unified pipeline.
"""

import time
import logging

from document_loader import load_documents_from_sources, fetch_evaluation_pairs
from text_chunker import process_and_segment_all
from embedding_engine import initialize_vector_model, generate_chunk_embeddings, encode_search_query
from vector_store import create_faiss_store, retrieve_nearest_neighbors, persist_vector_index, restore_vector_index
from hybrid_retriever import KeywordSearchEngine, combined_hybrid_retrieve, rerank_candidates
from answer_generator import produce_grounded_response
import config

app_logger = logging.getLogger(__name__)


class InformationRetrievalPipeline:
    """
    Retrieval-Augmented Generation pipeline orchestration class.
    Stores models, index state, and performance metrics across calls.
    """

    def __init__(self):
        self.encoder_model = None
        self.faiss_db = None
        self.lexical_engine = None
        self.segment_registry = []

        # System performance tracking dictionary
        self.metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_dimensions": 0,
            "chunk_size_setting": config.SEGMENT_LIMIT,
            "chunk_overlap_setting": config.SEGMENT_STRIDE,
            "embedding_model": config.ENCODER_MODEL_NAME,
            "llm_model": config.GENERATOR_MODEL,
            "faiss_index_type": "IndexFlatIP (inner product index)",
            "top_k": config.RETRIEVAL_LIMIT,
            "ingestion_time_sec": 0,
            "queries_answered": 0,
        }

    def load_and_index_documents(self, data_path=None, load_hf=False):
        """
        Runs the end-to-end ingestion pipeline:
        1. Loads raw documents from directory or HF.
        2. Breaks down documents into overlapping chunks.
        3. Encodes text chunks into dense vector representations.
        4. Configures the FAISS index database.
        5. Builds the lexical index (BM25) for hybrid retrieval.
        """
        start_time = time.time()

        if data_path is None:
            data_path = config.LOCAL_DOCS_PATH

        # Step 1: Load inputs
        app_logger.info("=" * 60)
        app_logger.info("PHASE 1: Loading raw documents")
        app_logger.info("=" * 60)

        raw_documents = load_documents_from_sources(
            target_path=data_path,
            load_hf=load_hf,
            hf_repo=config.REMOTE_DATA_SOURCE,
            hf_limit=config.MAX_REMOTE_DOCS
        )

        if not raw_documents:
            app_logger.error("No source documents detected. Check configuration and paths.")
            return False

        self.metrics["total_documents"] = len(raw_documents)

        # Step 2: Chunk text files
        app_logger.info("=" * 60)
        app_logger.info("PHASE 2: Splitting documents into overlapping segments")
        app_logger.info("=" * 60)

        self.segment_registry = process_and_segment_all(
            raw_documents,
            chunk_limit=config.SEGMENT_LIMIT,
            stride_limit=config.SEGMENT_STRIDE
        )

        self.metrics["total_chunks"] = len(self.segment_registry)

        if not self.segment_registry:
            app_logger.error("Text chunking process yielded 0 segments.")
            return False

        # Step 3: Embed document chunks
        app_logger.info("=" * 60)
        app_logger.info("PHASE 3: Computing vector representation embeddings")
        app_logger.info("=" * 60)

        self.encoder_model = initialize_vector_model(config.ENCODER_MODEL_NAME)
        vector_matrix = generate_chunk_embeddings(self.encoder_model, self.segment_registry)

        self.metrics["embedding_dimensions"] = vector_matrix.shape[1] if vector_matrix.size > 0 else 0

        # Step 4: Index vector embeddings
        app_logger.info("=" * 60)
        app_logger.info("PHASE 4: Organizing vector indexing database")
        app_logger.info("=" * 60)

        self.faiss_db = create_faiss_store(vector_matrix)

        # Persist index to file system
        persist_vector_index(self.faiss_db, config.VEC_INDEX_FILE)

        # Step 5: Lexical index compilation
        app_logger.info("=" * 60)
        app_logger.info("PHASE 5: Compiling lexical BM25 database")
        app_logger.info("=" * 60)
        self.lexical_engine = KeywordSearchEngine(self.segment_registry)

        elapsed_time = time.time() - start_time
        self.metrics["ingestion_time_sec"] = round(elapsed_time, 2)

        app_logger.info(f"Ingestion pipeline execution finished in {elapsed_time:.2f} seconds")
        return True

    def query_pipeline(self, search_phrase):
        """
        Retrieves context segments corresponding to the query phrase, 
        constructs a prompt, and requests a grounded answer from the LLM.
        """
        if self.faiss_db is None or self.encoder_model is None:
            return {
                "answer": "[ERROR] Information retrieval pipeline has not been initialized. Please run load_and_index_documents first.",
                "retrieved_chunks": [],
                "question": search_phrase
            }

        # Vectorize the query text
        encoded_query = encode_search_query(self.encoder_model, search_phrase)

        # Retrieve nearest neighbor context blocks
        if config.ENABLE_HYBRID_RETRIEVAL and self.lexical_engine is not None:
            top_candidates = combined_hybrid_retrieve(
                faiss_db=self.faiss_db,
                lexical_engine=self.lexical_engine,
                search_vector=encoded_query,
                transformer_model=self.encoder_model,
                search_query=search_phrase,
                segment_list=self.segment_registry,
                num_results=config.RETRIEVAL_LIMIT,
                weight_factor=config.BLENDING_FACTOR
            )
        else:
            top_candidates = retrieve_nearest_neighbors(
                self.faiss_db,
                encoded_query,
                self.segment_registry,
                num_results=config.RETRIEVAL_LIMIT
            )

        # Apply CrossEncoder re-ranking if configured
        if config.ENABLE_CE_RERANK:
            top_candidates = rerank_candidates(search_phrase, top_candidates, num_results=config.RETRIEVAL_LIMIT)

        # Generate a model response grounded in the context
        generation_output = produce_grounded_response(
            query_text=search_phrase,
            retrieved_segments=top_candidates,
            key_string=config.LLM_API_KEY,
            model_id=config.GENERATOR_MODEL
        )

        self.metrics["queries_answered"] += 1

        return {
            "answer": generation_output["answer"],
            "question": search_phrase,
            "retrieved_chunks": top_candidates,
            "prompt_length": generation_output["prompt_length"],
            "context_chunks_used": generation_output["context_chunks_used"],
            "model_used": generation_output["model"]
        }

    def execute_validation_run(self, eval_questions=None):
        """
        Runs validation queries to test system responses.
        Loads queries from HF datasets if none are provided.
        """
        evaluation_outputs = []

        if eval_questions is None:
            retrieved_qa_pairs = fetch_evaluation_pairs(config.REMOTE_DATA_SOURCE)
            if retrieved_qa_pairs:
                eval_questions = [pair["question"] for pair in retrieved_qa_pairs[:10]]
                app_logger.info(f"Loaded {len(eval_questions)} validation queries from dataset snapshot")
            else:
                # Default validation queries if HF load fails
                eval_questions = [
                    "What is supervised learning?",
                    "How does overfitting affect model performance?",
                    "What is the bias-variance tradeoff?",
                    "Explain transfer learning and why it is useful.",
                    "What are common evaluation metrics for classification?",
                ]
                app_logger.info("Applying default fallback validation queries")

        app_logger.info("=" * 60)
        app_logger.info(f"VALIDATION STAGE: Processing {len(eval_questions)} validation queries")
        app_logger.info("=" * 60)

        for pos, eval_question in enumerate(eval_questions, 1):
            app_logger.info(f"\n--- Processing Query {pos}/{len(eval_questions)} ---")
            app_logger.info(f"Query text: {eval_question}")

            pipeline_output = self.query_pipeline(eval_question)
            evaluation_outputs.append(pipeline_output)

            if pipeline_output["retrieved_chunks"]:
                highest_sim_score = pipeline_output["retrieved_chunks"][0]["relevance_score"]
                mean_sim_score = sum(c["relevance_score"] for c in pipeline_output["retrieved_chunks"]) / len(pipeline_output["retrieved_chunks"])
                app_logger.info(f"Retrieval scoring: top_score={highest_sim_score:.4f}, average_score={mean_sim_score:.4f}")

            short_response = pipeline_output["answer"][:200] + "..." if len(pipeline_output["answer"]) > 200 else pipeline_output["answer"]
            app_logger.info(f"Answer: {short_response}")

        return evaluation_outputs

    def generate_metrics_summary(self):
        """
        Produces a formatted metrics summary of database details, settings, and performance indicators.
        """
        summary_lines = [
            "",
            "=" * 65,
            "                      SYSTEM STATUS REPORT",
            "=" * 65,
            "",
            "--- Ingestion Details ---",
            f"  Documents processed:     {self.metrics['total_documents']}",
            f"  Created chunks count:    {self.metrics['total_chunks']}",
            f"  Ingestion duration:      {self.metrics['ingestion_time_sec']}s",
            "",
            "--- Segmentation Profile ---",
            f"  Max chunk size:          {self.metrics['chunk_size_setting']} characters",
            f"  Chunk stride length:     {self.metrics['chunk_overlap_setting']} characters",
            f"  Split algorithm:         Recursive Delimiter Strategy",
            "",
            "--- Vector Encoding Config ---",
            f"  Model identifier:        {self.metrics['embedding_model']}",
            f"  Vector features size:    {self.metrics['embedding_dimensions']} dims",
            f"  Normalization standard:  L2 scale",
            "",
            "--- Retrieval Execution ---",
            f"  Search engine:           FAISS index database + BM25 keyword",
            f"  FAISS Index schema:      {self.metrics['faiss_index_type']}",
            f"  Stored vector arrays:    {self.metrics['total_chunks']}",
            f"  Hybrid retrieval mode:   {'Enabled (alpha=' + str(config.BLENDING_FACTOR) + ')' if config.ENABLE_HYBRID_RETRIEVAL else 'Disabled (pure semantic vector)'}",
            f"  Re-ranking layer:        {'Enabled (ms-marco-MiniLM model)' if config.ENABLE_CE_RERANK else 'Disabled'}",
            f"  Top matches limit:       {self.metrics['top_k']}",
            "",
            "--- Language Model Endpoint ---",
            f"  API Provider:            Google Gemini",
            f"  Model variant:           {self.metrics['llm_model']}",
            f"  Addressed queries:       {self.metrics['queries_answered']}",
            "",
            "=" * 65,
        ]

        return "\n".join(summary_lines)
