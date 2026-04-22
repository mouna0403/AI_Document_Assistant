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
    "brief": "Write a very brief summary (1-2 sentences) of the following text:\n\n{context}",
    "standard": "Write a clear and concise summary of the following text:\n\n{context}",
    "detailed": "Write a detailed summary of the following text, including all key points:\n\n{context}",
}

load_dotenv()


def summarize_text(text: str, level: str = "standard") -> str:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

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
                chunk_size=3000, chunk_overlap=300
            )

            chunks = splitter.split_text(text)

            partial_summaries = []

            for chunk in chunks:
                try:
                    partial_summaries.append(run(chunk))
                except Exception:
                    continue

            combined = "\n\n".join(partial_summaries)

            try:
                return run(combined)
            except Exception:
                return combined

        raise e


if __name__ == "__main__":
    level = input("Choose your level: ")
    user_text = input("Put your text: ")
    print("------------------------------" * 50)
    print(summarize_text(user_text, level))
