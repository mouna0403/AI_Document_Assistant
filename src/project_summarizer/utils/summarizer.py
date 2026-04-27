"""
summarizer.py
LLM summarization with Groq Llama 3.1
- Primary: stuff_documents_chain
- Fallback: chunking only on 413/token overflow
- Recursive merge: handles 413 on merge step
"""

import logging
import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO)


# ---------------- PROMPTS ----------------

SUMMARY_PROMPTS = {
    "brief": """
You are a summarization engine.

Constraints:
- Max 2 sentences
- Keep only the main idea
- Remove examples and secondary details
- No repetition
- Use your own words

Output:
- 1–2 sentences only

Text:
{context}
""",
    "standard": """
You are a professional summarizer.

Constraints:
- 6 to 10 sentences total
- Keep only core ideas and reasoning
- Remove redundancy
- Ignore examples and minor details
- Use your own words

Output:
- 2 to 3 short paragraphs

Text:
{context}
""",
    "detailed": """
You are a professional summarizer.

Constraints:
- 12 to 18 sentences total
- Keep key concepts and reasoning
- Remove redundancy completely
- Preserve logical flow

Output:
- 3 to 5 paragraphs

Text:
{context}
""",
}


MERGE_PROMPTS = {
    "brief": """
You are merging multiple summaries into ONE unified summary.

Constraints:
- Max 2 sentences TOTAL
- Produce ONE single coherent summary
- Do NOT list or separate ideas
- Compress everything into one core idea
- No redundancy

Text:
{context}
""",
    "standard": """
You are merging multiple summaries into a coherent synthesis.

Constraints:
- 6 to 10 sentences total
- Merge overlapping ideas
- Remove redundancy
- Produce unified narrative

Text:
{context}
""",
    "detailed": """
You are merging multiple summaries into a structured synthesis.

Constraints:
- 12 to 18 sentences total
- Preserve key reasoning
- Remove redundancy
- Maintain logical flow

Text:
{context}
""",
}


# ---------------- LLM ----------------


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


# ---------------- CORE RUNNER ----------------


def run_chain(llm, prompt_text, text_input):
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = create_stuff_documents_chain(llm, prompt)

    doc = Document(page_content=text_input)

    result = chain.invoke({"context": [doc]})

    # robust extraction (LangChain version safe)
    if isinstance(result, dict):
        return result.get("output_text", "").strip()

    return str(result).strip()


def _is_overflow(err: str) -> bool:
    return "413" in err or "token" in err.lower() or "too large" in err.lower()


# ---------------- RECURSIVE MERGE ----------------


def recursive_merge(
    llm, partials: list[str], merge_prompt: str, level: str, depth: int = 0
) -> str:
    """
    Merge a list of partial summaries, re-chunking recursively if the
    combined text is still too large for the model.
    """
    if depth > 10:
        # Safety valve: just return the concatenation rather than infinite recursion
        logging.warning("Max merge depth reached — returning concatenation")
        return "\n\n".join(partials)

    combined = "\n\n".join(partials)

    try:
        logging.info(f"Merging {len(partials)} partial(s) (depth={depth})")
        return run_chain(llm, merge_prompt, combined)

    except Exception as e:
        err = str(e)
        logging.warning(f"Merge failed at depth {depth}: {err}")

        if not _is_overflow(err):
            raise e

        # Split partials into two halves and merge each half first
        if len(partials) == 1:
            # Single partial is still too large — summarise it with a tighter prompt
            logging.info("Single partial too large — summarising it further")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "],
            )
            sub_chunks = splitter.split_text(partials[0])
            sub_partials = []
            for i, chunk in enumerate(sub_chunks):
                try:
                    res = run_chain(llm, merge_prompt, chunk)
                    if res:
                        sub_partials.append(res)
                except Exception as ce:
                    logging.warning(f"Sub-chunk {i} failed: {ce}")
            if not sub_partials:
                return partials[0]  # give up gracefully
            return recursive_merge(llm, sub_partials, merge_prompt, level, depth + 1)

        mid = len(partials) // 2
        left = recursive_merge(llm, partials[:mid], merge_prompt, level, depth + 1)
        right = recursive_merge(llm, partials[mid:], merge_prompt, level, depth + 1)
        return recursive_merge(llm, [left, right], merge_prompt, level, depth + 1)


# ---------------- MAIN FUNCTION ----------------


def summarize_text(text: str, level: str = "standard") -> str:
    llm = get_llm()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    prompt_text = SUMMARY_PROMPTS[level]

    try:
        logging.info("Trying STUFF chain (no chunking)")
        return run_chain(llm, prompt_text, text)

    except Exception as e:
        err = str(e)
        logging.warning(f"STUFF failed: {err}")

        if not _is_overflow(err):
            raise e

        logging.info("Falling back to chunking")

        chunks = splitter.split_text(text)
        partials = []

        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")
                res = run_chain(llm, prompt_text, chunk)
                if res:
                    partials.append(res)
            except Exception as ce:
                logging.warning(f"Chunk {i} failed: {ce}")

        if not partials:
            return "Error: all chunks failed during summarization."

        merge_prompt = MERGE_PROMPTS[level]
        return recursive_merge(llm, partials, merge_prompt, level)


# ---------------- CLI ----------------

if __name__ == "__main__":
    level = input("Level (brief / standard / detailed): ")
    text = input("Text: ")

    print("\n" + "-" * 80 + "\n")
    print(summarize_text(text, level))
