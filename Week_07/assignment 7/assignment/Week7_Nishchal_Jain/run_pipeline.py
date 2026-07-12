"""
run_pipeline.py
---------------
Main orchestrator script to initialize logging, parse arguments,
trigger document ingestion, run demonstration queries, and print metrics.
"""

import sys
import logging
import argparse
from rag_pipeline import InformationRetrievalPipeline
import config


def configure_logger():
    """
    Initializes system logging. Logs everything to local files and filters
    INFO log level to the console output stream.
    """
    import os
    os.makedirs(config.LOGS_FOLDER, exist_ok=True)

    parent_logger = logging.getLogger()
    parent_logger.setLevel(logging.DEBUG)

    # Console display output configuration
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter("%(levelname)s | %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    # Debug file logging configuration
    logfile_handler = logging.FileHandler(config.LOG_OUTPUT_PATH, mode="w", encoding="utf-8")
    logfile_handler.setLevel(logging.DEBUG)
    logfile_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    logfile_handler.setFormatter(logfile_formatter)

    parent_logger.addHandler(stdout_handler)
    parent_logger.addHandler(logfile_handler)

    logging.info(f"Logging initialized. Output target file: {config.LOG_OUTPUT_PATH}")


def parse_command_line_flags():
    """
    Sets up the argument parsing configuration for execution flags.
    """
    arg_parser = argparse.ArgumentParser(
        description="Information Retrieval-Augmented Generation (RAG) System"
    )
    arg_parser.add_argument(
        "--hf",
        action="store_true",
        help="Include and parse files from the HuggingFace vectara/open_ragbench dataset"
    )
    arg_parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Specify local folder containing target documents (Default: sample_docs/)"
    )
    arg_parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Provide path to a single document to index instead of a directory"
    )
    arg_parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Run query search on a single query and quit"
    )
    arg_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enables CrossEncoder model re-ranking for results search matching"
    )
    arg_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Bypasses hybrid search mode to execute pure FAISS database lookups"
    )
    return arg_parser.parse_args()


def draw_separator(header_title=""):
    """Visual divider block generator for CLI outputs."""
    if header_title:
        print(f"\n{'='*65}")
        print(f"  {header_title}")
        print(f"{'='*65}")
    else:
        print(f"\n{'-'*65}")


def execute_sample_queries(retrieval_pipeline):
    """
    Performs standard queries to demonstrate capabilities of the retrieval system.
    """
    sample_queries = [
        "What is supervised learning and how does it differ from unsupervised learning?",
        "What techniques can prevent overfitting in machine learning models?",
        "Explain the bias-variance tradeoff.",
        "What is transfer learning and why is it important?",
        "How are machine learning models evaluated for classification tasks?",
    ]

    draw_separator("SAMPLE QUERIES: Validating Pipeline Responses")

    for counter, query_phrase in enumerate(sample_queries, 1):
        print(f"\n{'-'*50}")
        print(f"  Query {counter}: {query_phrase}")
        print(f"{'-'*50}")

        query_output = retrieval_pipeline.query_pipeline(query_phrase)

        print(f"\n  Response:\n  {query_output['answer']}")

        print(f"\n  Matched {query_output['context_chunks_used']} source chunks:")
        for matched_segment in query_output["retrieved_chunks"][:3]:
            sim_score = matched_segment["relevance_score"]
            doc_origin = matched_segment["source"]
            text_preview = matched_segment["chunk_text"][:100] + "..." if len(matched_segment["chunk_text"]) > 100 else matched_segment["chunk_text"]
            print(f"    [{sim_score:.3f}] ({doc_origin}) {text_preview}")

        print()


def main():
    configure_logger()
    parsed_flags = parse_command_line_flags()

    main_logger = logging.getLogger(__name__)

    if parsed_flags.rerank:
        config.ENABLE_CE_RERANK = True
    if parsed_flags.no_hybrid:
        config.ENABLE_HYBRID_RETRIEVAL = False

    draw_separator("IR-RAG Pipeline - Starting Process")
    print(f"  Encoding Model Name:    {config.ENCODER_MODEL_NAME}")
    print(f"  Generative Model:       {config.GENERATOR_MODEL}")
    print(f"  Max Character Size:     {config.SEGMENT_LIMIT} (Stride Overlap: {config.SEGMENT_STRIDE})")
    print(f"  Retrieval Type Mode:    {'Hybrid Lexical + Semantic' if config.ENABLE_HYBRID_RETRIEVAL else 'Pure Semantic Vector'}")
    print(f"  Reranking Layer Status: {'Enabled (CrossEncoder Model)' if config.ENABLE_CE_RERANK else 'Disabled'}")
    print(f"  HuggingFace dataset:    {'Enabled' if parsed_flags.hf else 'Disabled'}")

    if not config.LLM_API_KEY:
        print("\n  WARNING: Environment variable GEMINI_API_KEY is not defined!")
        print("  The retrieval pipeline runs but answer generation returns an error.")
        print("  Access free keys from: https://aistudio.google.com/")
        print("  Configure by setting keys inside a local .env configuration file.")

    retrieval_pipeline = InformationRetrievalPipeline()

    draw_separator("Phase 1: Document Processing and Indexing")

    if parsed_flags.file:
        ingestion_path = parsed_flags.file
        print(f"  Target File:     {parsed_flags.file}")
    elif parsed_flags.docs_dir:
        ingestion_path = parsed_flags.docs_dir
        print(f"  Target Folder:   {parsed_flags.docs_dir}")
    else:
        ingestion_path = config.LOCAL_DOCS_PATH
        print(f"  Target Folder:   {config.LOCAL_DOCS_PATH}")

    is_ingested = retrieval_pipeline.load_and_index_documents(
        data_path=ingestion_path,
        load_hf=parsed_flags.hf
    )

    if not is_ingested:
        print("\n  Pipeline Ingestion failed. Check log outputs for trace information.")
        print(f"  Logs saved: {config.LOG_OUTPUT_PATH}")
        sys.exit(1)

    print(f"\n  Indexing completed successfully!")
    print(f"  Documents Loaded: {retrieval_pipeline.metrics['total_documents']}")
    print(f"  Segments Created: {retrieval_pipeline.metrics['total_chunks']}")
    print(f"  Ingest Duration:  {retrieval_pipeline.metrics['ingestion_time_sec']}s")

    # Mode branch selection
    if parsed_flags.question:
        draw_separator("Single Query Mode Execution")
        print(f"\n  Query: {parsed_flags.question}\n")

        query_output = retrieval_pipeline.query_pipeline(parsed_flags.question)
        print(f"  Response: {query_output['answer']}")

        print(f"\n  Matched segments:")
        for matched_segment in query_output["retrieved_chunks"]:
            sim_score = matched_segment["relevance_score"]
            doc_origin = matched_segment["source"]
            print(f"    [{sim_score:.3f}] ({doc_origin})")
    else:
        # Run demo evaluation queries
        execute_sample_queries(retrieval_pipeline)

        # Run system validation step
        draw_separator("Phase 2: System Validation Queries")
        eval_results = retrieval_pipeline.execute_validation_run()

        print(f"\n  Validation finished: {len(eval_results)} queries processed")
        if eval_results:
            top_similarity_scores = []
            for result_record in eval_results:
                if result_record["retrieved_chunks"]:
                    top_similarity_scores.append(result_record["retrieved_chunks"][0]["relevance_score"])

            if top_similarity_scores:
                average_top_score = sum(top_similarity_scores) / len(top_similarity_scores)
                minimum_top_score = min(top_similarity_scores)
                maximum_top_score = max(top_similarity_scores)
                print(f"  Vector alignment matches (Top-1): average={average_top_score:.4f}, min={minimum_top_score:.4f}, max={maximum_top_score:.4f}")

    # Generate print report outputs
    metrics_report = retrieval_pipeline.generate_metrics_summary()
    print(metrics_report)

    print(f"\n  Executing logs: {config.LOG_OUTPUT_PATH}")
    print(f"  Database Index:  {config.VEC_INDEX_FILE}")
    print()


if __name__ == "__main__":
    main()
