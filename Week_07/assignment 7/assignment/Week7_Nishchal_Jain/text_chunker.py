"""
text_chunker.py
---------------
Splits large documents recursively into smaller segments. Preserves paragraph,
sentence, and word boundaries to minimize semantic context loss.
"""

import logging

app_logger = logging.getLogger(__name__)


def _locate_split_boundary(content, delimiters):
    """
    Finds the first delimiter in the prioritized hierarchy that is present in the text.
    """
    for delim in delimiters:
        if delim in content:
            return delim
    return None


def _partition_text_recursively(content, max_size, delimiters):
    """
    Recursively splits text using the hierarchy of delimiters until
    every segment fits within the specified character limit.
    """
    # Return immediately if the text is small enough
    if len(content) <= max_size:
        return [content]

    optimal_delim = _locate_split_boundary(content, delimiters)

    # If no delimiter is found, enforce a hard character split
    if optimal_delim is None:
        sub_blocks = []
        for offset in range(0, len(content), max_size):
            sub_blocks.append(content[offset:offset + max_size])
        return sub_blocks

    # Split using the discovered delimiter
    tokens = content.split(optimal_delim)

    # Combine small tokens up to the character limit
    assembled_chunks = []
    accumulated_str = ""

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        candidate_merge = accumulated_str + optimal_delim + token if accumulated_str else token

        if len(candidate_merge) <= max_size:
            accumulated_str = candidate_merge
        else:
            if accumulated_str:
                assembled_chunks.append(accumulated_str)
            
            # Recurse if a single token exceeds the character limit
            if len(token) > max_size:
                fallback_delims = delimiters[delimiters.index(optimal_delim) + 1:] if optimal_delim in delimiters else []
                deeper_chunks = _partition_text_recursively(token, max_size, fallback_delims)
                assembled_chunks.extend(deeper_chunks)
                accumulated_str = ""
            else:
                accumulated_str = token

    if accumulated_str:
        assembled_chunks.append(accumulated_str)

    return assembled_chunks


def segment_text(input_text, max_len=500, overlap_len=80, doc_origin="unknown"):
    """
    Segments the input text into a list of dictionaries containing text, index,
    and source metadata. Applies an overlap between adjacent segments.
    """
    if not input_text or not input_text.strip():
        return []

    sanitized_input = input_text.strip()

    # Prioritize breaks at paragraphs, then sentences, and finally punctuation/words
    delimiters_hierarchy = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    unprocessed_segments = _partition_text_recursively(sanitized_input, max_len, delimiters_hierarchy)

    # Implement sliding window overlap between consecutive chunks
    segments_with_context = []
    for pos, segment in enumerate(unprocessed_segments):
        if pos > 0 and overlap_len > 0:
            trailing_context = unprocessed_segments[pos - 1][-overlap_len:]
            boundary_idx = trailing_context.find(" ")
            if boundary_idx != -1:
                trailing_context = trailing_context[boundary_idx + 1:]
            segment = trailing_context + " " + segment

        segments_with_context.append(segment.strip())

    # Build metadata payloads
    metadata_chunks = []
    for pos, segment_text_content in enumerate(segments_with_context):
        # Exclude extremely short noise fragments
        if segment_text_content and len(segment_text_content) > 10:
            metadata_chunks.append({
                "chunk_text": segment_text_content,
                "chunk_index": pos,
                "source": doc_origin
            })

    app_logger.info(
        f"Segmented '{doc_origin}' into {len(metadata_chunks)} blocks "
        f"(max_size={max_len}, overlap={overlap_len})"
    )
    return metadata_chunks


def process_and_segment_all(doc_list, chunk_limit=500, stride_limit=80):
    """
    Wrapper to segment an entire batch of documents.
    """
    aggregated_chunks = []

    for document in doc_list:
        document_segments = segment_text(
            input_text=document["text"],
            max_len=chunk_limit,
            overlap_len=stride_limit,
            doc_origin=document["source"]
        )
        aggregated_chunks.extend(document_segments)

    app_logger.info(f"Segmentation finalized: {len(aggregated_chunks)} chunks from {len(doc_list)} source files")
    return aggregated_chunks
