"""
answer_generator.py
-------------------
Implements prompt engineering and invokes the Google Gemini API using the modern
google-genai SDK. Guarantees grounded answers by enforcing instructions.
"""

import logging

app_logger = logging.getLogger(__name__)


def construct_context_prompt(query_text, retrieved_segments):
    """
    Assembles a system-instructed prompt containing context segments, metadata, and the question.
    """
    aggregated_context = ""
    for counter, segment in enumerate(retrieved_segments, 1):
        origin = segment.get("source", "unknown")
        confidence_score = segment.get("relevance_score", 0.0)
        text_val = segment.get("chunk_text", "")
        aggregated_context += f"\n--- Section {counter} (Origin: {origin}, Confidence: {confidence_score:.3f}) ---\n"
        aggregated_context += text_val + "\n"

    formatted_prompt = f"""You are an assistant. Answer the question using ONLY the provided contexts.

INSTRUCTIONS:
- Base your answer exclusively on the given contexts below
- If the contexts do not contain enough facts to resolve the query, state that clearly
- Provide clear and concise responses
- Cite which sections were used to build the answer if applicable

CONTEXTS:
{aggregated_context}

USER QUESTION: {query_text}

ANSWER:"""

    return formatted_prompt


def invoke_gemini_endpoint(formatted_prompt, key_string, model_id="gemini-flash-latest"):
    """
    Sends the constructed query payload to Google's Gemini generative service.
    Handles credential validation and API level exceptions.
    """
    key_string = key_string.strip() if key_string else ""
    if not key_string:
        failure_detail = (
            "Gemini API key is not configured. Retrieve a key at "
            "https://aistudio.google.com/ and declare it in your .env environment file."
        )
        app_logger.error(failure_detail)
        return f"[ERROR] {failure_detail}"

    try:
        from google import genai
    except ImportError:
        app_logger.error("google-genai library is missing. Install using: pip install google-genai")
        return "[ERROR] google-genai package is not installed in environment"

    try:
        gemini_client = genai.Client(api_key=key_string)

        api_response = gemini_client.models.generate_content(
            model=model_id,
            contents=formatted_prompt
        )

        generated_text = api_response.text.strip()
        app_logger.info(f"Gemini API execution succeeded. Length: {len(generated_text)} characters")
        return generated_text

    except Exception as api_err:
        failure_detail = f"Gemini API endpoint failed: {str(api_err)}"
        app_logger.error(failure_detail)
        return f"[ERROR] {failure_detail}"


def produce_grounded_response(query_text, retrieved_segments, key_string, model_id="gemini-2.0-flash"):
    """
    Main entry point for generating answers. Constructs prompt and requests response from API.
    """
    formatted_prompt = construct_context_prompt(query_text, retrieved_segments)

    app_logger.info(f"Sending request to Gemini model ({len(formatted_prompt)} chars, {len(retrieved_segments)} segments)")
    generated_response = invoke_gemini_endpoint(formatted_prompt, key_string, model_id)

    response_payload = {
        "answer": generated_response,
        "prompt_length": len(formatted_prompt),
        "context_chunks_used": len(retrieved_segments),
        "model": model_id,
        "question": query_text
    }

    return response_payload
