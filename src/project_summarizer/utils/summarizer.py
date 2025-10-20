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

# Mapping of summary levels to instructions
SUMMARY_PROMPTS = {
    "brief": "Write a very brief summary (1-2 sentences) of the following text:\n\n{context}",
    "standard": "Write a clear and concise summary of the following text:\n\n{context}",
    "detailed": "Write a detailed summary of the following text, including all key points:\n\n{context}",
}

load_dotenv()


def summarize_text(text: str, level) -> str:
    """Summarize the provided text using a Groq-hosted Llama model."""

    # Initialize the LLM (ensure the API key is set in environment variables)
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

    # Define the prompt based on the selected summary level
    prompt_text = SUMMARY_PROMPTS[level]
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])

    # Create summarization chain
    chain = create_stuff_documents_chain(llm, prompt)

    # Convert text to Document object
    doc = Document(page_content=text)

    # Run summarization
    result = chain.invoke({"context": [doc]})

    return result


if __name__ == "__main__":
    user_text = input("Put your text : ")
    print(summarize_text(user_text))
