"""
summarizer.py
This module handles LLM-based summarization of document text using Groq's Llama 3.1 model.
Supports multi-level summaries: brief, standard, detailed.
"""

import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUMMARY_PROMPTS = {
    "brief": """
You are a summarization engine.

Constraints:
- Max 2 sentences
- Use your own words only — never copy phrases from the source
- No repetition
- Focus only on the main argument, ignore examples and secondary points
- Do not introduce any external concept

Text:
{context}
""",
    "standard": """
You are a professional summarizer.

Constraints:
- In 8–12 lines max
- Use your own words only — never copy phrases from the source
- Remove all redundancy
- Focus only on the main argument, ignore examples and secondary points
- Preserve logical flow
- Do not introduce any concept not present in the text

Format:
- Bullet points only

Text:
{context}
""",
    "detailed": """
Produce a structured summary.

Constraints:
- In 12–18 lines max
- Use your own words only — never copy phrases from the source
- No repetition
- Focus on key concepts only (ignore examples, anecdotes, minor details)
- Do not introduce external knowledge

Structure:
1. Main idea
2. Key concepts
3. Mechanisms / reasoning
4. Key distinctions
5. Conclusion

Text:
{context}
""",
}

MERGE_PROMPTS = {
    "brief": """
You are merging multiple partial summaries.

Constraints:
- Max 2 sentences
- Keep only the core idea
- Remove all redundancy
- Use your own words only — never copy phrases
- Do not introduce any new concept

Text:
{context}
""",
    "standard": """
You are merging multiple partial summaries.

Constraints:
- In 8–12 lines max
- Remove duplicates completely
- Keep only essential ideas
- Use your own words only — never copy phrases
- Reorganize logically (general → specific)
- Do not introduce any new concept

Format:
- Bullet points only

Text:
{context}
""",
    "detailed": """
You are merging multiple partial summaries.

Constraints:
- In 12–18 lines max
- Remove redundancy
- Keep all important concepts but no repetition
- Use your own words only — never copy phrases
- Reorganize into a clear structure
- Do not introduce any new concept

Structure:
1. Main idea
2. Key concepts
3. Mechanisms / reasoning
4. Key distinctions
5. Conclusion

Text:
{context}
""",
}

load_dotenv()


def summarize_text(text: str, level: str = "standard") -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    prompt_text = SUMMARY_PROMPTS[level]
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])

    chain = create_stuff_documents_chain(llm, prompt)

    def run(text_input: str) -> str:
        doc = Document(page_content=text_input)
        return chain.invoke({"context": [doc]})

    try:
        # First attempt: full text
        return run(text)

    except Exception as e:
        err = str(e)

        # Fallback only for token/size errors
        if "413" in err or "tokens" in err or "too large" in err:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", " "],
            )

            chunks = splitter.split_text(text)

            partial_summaries = []

            for chunk in chunks:
                try:
                    partial_summaries.append(run(chunk))
                except Exception:
                    continue

            combined = "\n\n".join(partial_summaries)

            merge_prompt_text = MERGE_PROMPTS[level]
            merge_prompt = ChatPromptTemplate.from_messages(
                [("system", merge_prompt_text)]
            )
            merge_chain = create_stuff_documents_chain(llm, merge_prompt)

            try:
                doc = Document(page_content=combined)
                return merge_chain.invoke({"context": [doc]})
            except Exception:
                return combined

        raise e


if __name__ == "__main__":
    level = input("Choose your level: ")
    user_text = input("Put your text: ")
    print("------------------------------" * 50)
    print(summarize_text(user_text, level))
