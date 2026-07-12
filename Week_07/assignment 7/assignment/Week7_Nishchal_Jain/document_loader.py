"""
document_loader.py
------------------
Module responsible for importing text documents from different sources, including
local plain text files, PDF documents, and the HuggingFace dataset repo.
"""

import os
import json
import glob
import logging

app_logger = logging.getLogger(__name__)

# Cached local directory for the HuggingFace dataset to avoid redundant downloads
_DATASET_CACHE_DIR = None


def extract_pdf_content(pdf_path):
    """
    Reads a PDF document page by page and aggregates the text content.
    Uses PyPDF2 for parsing. Returns empty string if the package is missing.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        app_logger.error("PyPDF2 is not installed. Please run: pip install PyPDF2")
        return ""

    if not os.path.exists(pdf_path):
        app_logger.warning(f"PDF file not found at {pdf_path}, skipping text extraction")
        return ""

    pdf_reader = PdfReader(pdf_path)
    extracted_pages = []

    for idx, page_obj in enumerate(pdf_reader.pages):
        try:
            page_text = page_obj.extract_text()
            if page_text and page_text.strip():
                extracted_pages.append(page_text.strip())
                app_logger.debug(f"Successfully read {len(page_text)} characters from page {idx + 1}")
        except Exception as read_err:
            app_logger.warning(f"Error reading page {idx + 1} of {pdf_path}: {read_err}")
            continue

    full_text = "\n\n".join(extracted_pages)
    app_logger.info(f"Extracted {len(full_text)} characters from PDF file: {os.path.basename(pdf_path)}")
    return full_text


def read_local_text_file(txt_path):
    """
    Reads plain text or markdown files from the local filesystem.
    """
    if not os.path.exists(txt_path):
        app_logger.warning(f"Text file not found at {txt_path}, skipping")
        return ""

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text_data = f.read().strip()

    app_logger.info(f"Successfully loaded {len(text_data)} characters from {os.path.basename(txt_path)}")
    return text_data


def _fetch_dataset_cache_path(repo_id):
    """
    Resolves or downloads the local directory path where the HuggingFace
    dataset is cached. Returns the absolute directory path.
    """
    global _DATASET_CACHE_DIR
    if _DATASET_CACHE_DIR and os.path.isdir(_DATASET_CACHE_DIR):
        return _DATASET_CACHE_DIR

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        app_logger.error("huggingface_hub package is not installed. Run: pip install huggingface-hub")
        return None

    app_logger.info(f"Resolving/downloading dataset repository: {repo_id}")
    app_logger.info("This download may take a moment on the initial run.")

    try:
        cached_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset"
        )
        _DATASET_CACHE_DIR = cached_dir
        app_logger.info(f"Dataset snapshot found at: {cached_dir}")
        return cached_dir
    except Exception as fetch_err:
        app_logger.error(f"Error downloading the HuggingFace dataset: {fetch_err}")
        return None


def ingest_remote_dataset(repo_id, limit_docs=50):
    """
    Parses document structured JSON files from the HuggingFace dataset snapshot.
    Combines sections from each document file and returns them.
    """
    dataset_root = _fetch_dataset_cache_path(repo_id)
    if not dataset_root:
        return []

    docs_folder = os.path.join(dataset_root, "pdf", "arxiv", "corpus")
    if not os.path.isdir(docs_folder):
        app_logger.error(f"Corpus subdirectory not found at: {docs_folder}")
        return []

    json_filepaths = sorted(glob.glob(os.path.join(docs_folder, "*.json")))
    app_logger.info(f"Found {len(json_filepaths)} document files in the dataset")

    target_json_files = json_filepaths[:limit_docs]
    collected_docs = []

    for json_path in target_json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc_struct = json.load(f)

            doc_identifier = doc_struct.get("id", os.path.basename(json_path).replace(".json", ""))
            doc_title = doc_struct.get("title", "Untitled Document").strip()
            doc_sections = doc_struct.get("sections", [])

            parts_list = []
            for section in doc_sections:
                text_val = section.get("text", "").strip()
                if text_val:
                    parts_list.append(text_val)

            if parts_list:
                unified_text = f"{doc_title}\n\n" + "\n\n".join(parts_list)
                collected_docs.append({
                    "text": unified_text,
                    "source": f"arxiv:{doc_identifier}",
                    "doc_id": doc_identifier
                })

        except (json.JSONDecodeError, KeyError) as parse_err:
            app_logger.warning(f"Error parsing document structure in {os.path.basename(json_path)}: {parse_err}")
            continue

    app_logger.info(f"Successfully processed {len(collected_docs)} documents from dataset corpus")
    return collected_docs


def fetch_evaluation_pairs(repo_id, limit_pairs=30):
    """
    Loads query and gold standard response pairs from the HuggingFace dataset
    for system validation and testing.
    """
    dataset_root = _fetch_dataset_cache_path(repo_id)
    if not dataset_root:
        return []

    base_dataset_dir = os.path.join(dataset_root, "pdf", "arxiv")
    questions_json = os.path.join(base_dataset_dir, "queries.json")
    answers_json = os.path.join(base_dataset_dir, "answers.json")

    if not os.path.exists(questions_json) or not os.path.exists(answers_json):
        app_logger.warning("Queries or answers registry files not found in dataset snapshot")
        return []

    try:
        with open(questions_json, "r", encoding="utf-8") as f:
            questions_dict = json.load(f)

        with open(answers_json, "r", encoding="utf-8") as f:
            answers_dict = json.load(f)

        evaluation_records = []
        for entry_id, query_meta in questions_dict.items():
            if len(evaluation_records) >= limit_pairs:
                break

            query_string = query_meta.get("query", "")
            response_string = answers_dict.get(entry_id, "")

            if isinstance(response_string, dict):
                response_string = response_string.get("text", str(response_string))

            if query_string and response_string and len(query_string) > 10:
                evaluation_records.append({
                    "question": query_string.strip(),
                    "answer": str(response_string).strip(),
                    "query_type": query_meta.get("type", "unknown"),
                    "source_type": query_meta.get("source", "unknown")
                })

        app_logger.info(f"Loaded {len(evaluation_records)} query-response pairs for evaluation")
        return evaluation_records

    except Exception as load_err:
        app_logger.warning(f"Failed to load evaluation pairs: {load_err}")
        return []


def load_documents_from_sources(target_path, load_hf=False, hf_repo=None, hf_limit=50):
    """
    Aggregates text data from local path (single file or directory) and optionally
    appends documents downloaded from the HuggingFace repository.
    """
    all_loaded_docs = []

    if not target_path:
        app_logger.warning("Empty source path provided, skipping local file scans")
    elif os.path.isfile(target_path):
        app_logger.info(f"Attempting to read single file: {target_path}")
        file_name = os.path.basename(target_path)
        if file_name.lower().endswith(".pdf"):
            extracted_content = extract_pdf_content(target_path)
            if extracted_content:
                all_loaded_docs.append({"text": extracted_content, "source": file_name})
        elif file_name.lower().endswith((".txt", ".md")):
            extracted_content = read_local_text_file(target_path)
            if extracted_content:
                all_loaded_docs.append({"text": extracted_content, "source": file_name})
        else:
            app_logger.warning(f"File extension not supported for indexing: {file_name}")
    elif os.path.isdir(target_path):
        app_logger.info(f"Scanning target directory for documents: {target_path}")
        for file_name in sorted(os.listdir(target_path)):
            absolute_file_path = os.path.join(target_path, file_name)

            if file_name.lower().endswith(".pdf"):
                extracted_content = extract_pdf_content(absolute_file_path)
                if extracted_content:
                    all_loaded_docs.append({"text": extracted_content, "source": file_name})

            elif file_name.lower().endswith((".txt", ".md")):
                extracted_content = read_local_text_file(absolute_file_path)
                if extracted_content:
                    all_loaded_docs.append({"text": extracted_content, "source": file_name})

            else:
                app_logger.debug(f"Skipping unsupported file extension: {file_name}")
    else:
        app_logger.warning(f"The source folder or file path does not exist: {target_path}")

    if load_hf and hf_repo:
        remote_docs = ingest_remote_dataset(hf_repo, limit_docs=hf_limit)
        for doc in remote_docs:
            all_loaded_docs.append({
                "text": doc["text"],
                "source": doc["source"]
            })

    app_logger.info(f"Total documents loaded into pipeline: {len(all_loaded_docs)}")
    return all_loaded_docs
