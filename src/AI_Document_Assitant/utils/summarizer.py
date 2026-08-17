"""
summarizer.py
LLM summarization with Groq Llama 3.1

Improvements over v1:
- Direct Groq API calls (no LangChain chain wrapper) — full control over system/user prompt split
- Strict format constraints in prompts: no bullets, no headers, no lists, plain prose only
- Parallel chunk processing via ThreadPoolExecutor
- Chains/prompts built once and reused across all chunks
- Larger chunks (4000 chars) to minimize total API calls
- Retry with exponential backoff on transient errors (rate-limit, network)
- Recursive merge also runs in parallel where possible
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 3000  # chars — ~750 tokens, leaving room for prompts/context
CHUNK_OVERLAP = 200  # chars — preserves context between chunks
MAX_WORKERS = 2  # Groq free tier — safer with the 8,000 tokens/minute limit
MAX_RETRIES = 3  # retry count for transient failures (429, network)


# ---------------------------------------------------------------------------
# PROMPTS
# Two-part structure: system prompt (role + strict format rules) +
# user prompt (task + text).
# Keeping format rules in the SYSTEM message makes Llama 3.1 far more
# likely to honour them than burying them in a single user prompt.
# ---------------------------------------------------------------------------

# System prompts enforce the output format with explicit prohibitions.
# "ONLY plain prose" + "NEVER use" leaves no room for interpretation.
SYSTEM_PROMPTS = {
    "brief": (
        "You are a summarization engine. "
        "Your output MUST be plain prose: 1 to 2 sentences, nothing more. "
        "NEVER use bullet points, numbered lists, bold text, italic text, "
        "headers, section titles, or any markdown formatting. "
        "Write ONLY plain sentences."
    ),
    "standard": (
        "You are a professional summarizer. "
        "Your output MUST be plain prose organized in 2 to 3 short paragraphs "
        "totalling 6 to 10 sentences. "
        "NEVER use bullet points, numbered lists, bold text, italic text, "
        "headers, section titles, or any markdown formatting. "
        "Write ONLY plain paragraphs separated by a blank line."
    ),
    "detailed": (
        "You are a professional summarizer. "
        "Your output MUST be plain prose organized in 3 to 5 paragraphs "
        "totalling 12 to 18 sentences. "
        "NEVER use bullet points, numbered lists, bold text, italic text, "
        "headers, section titles, or any markdown formatting. "
        "Write ONLY plain paragraphs separated by a blank line."
    ),
}

# User prompts carry the task instruction + the text to process.
SUMMARY_USER_PROMPTS = {
    "brief": (
        "Summarize the following text in 1 to 2 plain sentences. "
        "Keep only the single most important idea. "
        "No lists, no headers, no formatting of any kind.\n\n"
        "Text:\n{text}"
    ),
    "standard": (
        "Summarize the following text in 2 to 3 plain paragraphs (6–10 sentences total). "
        "Keep only the core ideas. Remove all redundancy and minor details. "
        "No lists, no headers, no bold, no formatting of any kind. "
        "Separate paragraphs with a blank line.\n\n"
        "Text:\n{text}"
    ),
    "detailed": (
        "Summarize the following text in 3 to 5 plain paragraphs (12–18 sentences total). "
        "Preserve the key concepts and logical flow. Remove all redundancy. "
        "No lists, no headers, no bold, no formatting of any kind. "
        "Separate paragraphs with a blank line.\n\n"
        "Text:\n{text}"
    ),
}

MERGE_USER_PROMPTS = {
    "brief": (
        "The following are partial summaries of different sections of the same document. "
        "Merge them into ONE single summary of 1 to 2 plain sentences. "
        "Compress everything into the single most important idea. "
        "No lists, no headers, no formatting.\n\n"
        "Partial summaries:\n{text}"
    ),
    "standard": (
        "The following are partial summaries of different sections of the same document. "
        "Merge them into a unified summary of 2 to 3 plain paragraphs (6–10 sentences total). "
        "Eliminate all overlapping or redundant content. Produce one coherent narrative. "
        "No lists, no headers, no bold, no formatting of any kind. "
        "Separate paragraphs with a blank line.\n\n"
        "Partial summaries:\n{text}"
    ),
    "detailed": (
        "The following are partial summaries of different sections of the same document. "
        "Merge them into a unified summary of 3 to 5 plain paragraphs (12–18 sentences total). "
        "Preserve key reasoning and logical flow. Remove all redundancy. "
        "No lists, no headers, no bold, no formatting of any kind. "
        "Separate paragraphs with a blank line.\n\n"
        "Partial summaries:\n{text}"
    ),
}


# ---------------------------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------------------------


def get_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# CORE RUNNER — direct Groq API, no LangChain wrapper
# Using the chat completions endpoint directly gives us a clean
# system / user split, which is what makes the format constraints stick.
# Retry with exponential backoff handles 429 rate-limits and network blips.
# ---------------------------------------------------------------------------


def call_llm(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call the Groq chat completions endpoint with retry + exponential backoff.
    Returns the model's reply as a plain string.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_exc = e
            wait = 2**attempt  # 1s → 2s → 4s
            logging.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e} "
                f"— retrying in {wait}s"
            )
            time.sleep(wait)

    raise last_exc


def _is_overflow(err: str) -> bool:
    return "413" in err or "token" in err.lower() or "too large" in err.lower()


# ---------------------------------------------------------------------------
# PARALLEL CHUNK PROCESSING
# Each chunk is submitted as an independent future so all LLM calls
# run concurrently. The bottleneck is network I/O, not CPU, so
# ThreadPoolExecutor releases the GIL during each HTTP call.
# Results are reconstructed in original chunk order.
# ---------------------------------------------------------------------------


def process_chunks_parallel(
    client: Groq,
    system_prompt: str,
    user_prompt_template: str,
    chunks: list[str],
) -> list[str]:
    """
    Summarize each chunk concurrently.
    Returns partial summaries in the same order as the input chunks.
    Chunks that fail after all retries are skipped with a warning.
    """
    results: dict[int, str] = {}

    def process_one(idx: int, chunk: str) -> tuple[int, str]:
        user_prompt = user_prompt_template.format(text=chunk)
        result = call_llm(client, system_prompt, user_prompt)
        return idx, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_one, i, chunk): i for i, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                i, result = future.result()
                if result:
                    results[i] = result
                    logging.info(f"Chunk {i + 1}/{len(chunks)} done")
            except Exception as e:
                logging.warning(f"Chunk {idx} failed after all retries: {e}")

    return [results[i] for i in sorted(results)]


# ---------------------------------------------------------------------------
# RECURSIVE MERGE — parallel on overflow
# When the combined partials exceed the model context, split into
# two halves and merge each half concurrently before the final merge.
# ---------------------------------------------------------------------------


def recursive_merge(
    client: Groq,
    system_prompt: str,
    merge_prompt_template: str,
    partials: list[str],
    level: str,
    depth: int = 0,
) -> str:
    """
    Merge partial summaries, re-chunking recursively on token overflow.
    Left/right sub-merges run in parallel when splitting is needed.
    """
    if depth > 10:
        logging.warning("Max merge depth reached — returning concatenation")
        return "\n\n".join(partials)

    combined = "\n\n".join(partials)
    user_prompt = merge_prompt_template.format(text=combined)

    try:
        logging.info(f"Merging {len(partials)} partial(s) (depth={depth})")
        return call_llm(client, system_prompt, user_prompt)

    except Exception as e:
        err = str(e)
        logging.warning(f"Merge failed at depth {depth}: {err}")

        if not _is_overflow(err):
            raise e

        # Single oversized partial → sub-chunk and re-summarise
        if len(partials) == 1:
            logging.info("Single partial too large — sub-chunking it")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "],
            )
            sub_chunks = splitter.split_text(partials[0])
            sub_partials = process_chunks_parallel(
                client, system_prompt, merge_prompt_template, sub_chunks
            )
            if not sub_partials:
                return partials[0]
            return recursive_merge(
                client,
                system_prompt,
                merge_prompt_template,
                sub_partials,
                level,
                depth + 1,
            )

        # Split into two halves and merge each in parallel
        mid = len(partials) // 2
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_left = executor.submit(
                recursive_merge,
                client,
                system_prompt,
                merge_prompt_template,
                partials[:mid],
                level,
                depth + 1,
            )
            f_right = executor.submit(
                recursive_merge,
                client,
                system_prompt,
                merge_prompt_template,
                partials[mid:],
                level,
                depth + 1,
            )
            left = f_left.result()
            right = f_right.result()

        return recursive_merge(
            client,
            system_prompt,
            merge_prompt_template,
            [left, right],
            level,
            depth + 1,
        )


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------


def summarize_text(text: str, level: str = "standard") -> str:
    if level not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown level '{level}'. Choose: brief, standard, detailed.")

    client = get_client()
    system_prompt = SYSTEM_PROMPTS[level]
    summary_tpl = SUMMARY_USER_PROMPTS[level]
    merge_tpl = MERGE_USER_PROMPTS[level]

    # --- Attempt 1: single-shot (whole text, no chunking) ---
    try:
        logging.info("Trying single-shot summarization")
        user_prompt = summary_tpl.format(text=text)
        return call_llm(client, system_prompt, user_prompt)

    except Exception as e:
        err = str(e)
        logging.warning(f"Single-shot failed: {err}")
        if not _is_overflow(err):
            raise e

    # --- Fallback: parallel chunk processing ---
    logging.info("Falling back to parallel chunking")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    logging.info(f"Split into {len(chunks)} chunk(s)")

    partials = process_chunks_parallel(client, system_prompt, summary_tpl, chunks)

    if not partials:
        return "Error: all chunks failed during summarization."

    if len(partials) == 1:
        return partials[0]

    return recursive_merge(client, system_prompt, merge_tpl, partials, level)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    level = input("Level (brief / standard / detailed): ").strip() or "standard"
    text = input("Text: ").strip()
    print("\n" + "-" * 80 + "\n")
    print(summarize_text(text, level))
