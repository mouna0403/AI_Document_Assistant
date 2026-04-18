"""
qa_rag.py
Single-file RAG system: document indexing + retrieval + Groq QA with memory.
"""

import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def build_vectorstore(text: str):
    """
    Split document and build FAISS vector store using Ollama embeddings.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)

    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    # Local embeddings via Ollama
    embeddings = OllamaEmbeddings(model="llama3.2:3b")

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def create_qa(vectorstore):
    """
    Create QA system using retrieval + Groq LLM.
    """

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise and factual assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, respond exactly: "
                "'I cannot find this information in the provided document.'\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def answer_question(question: str):
        # chat_history = memory.load_memory_variables({})["chat_history"]

        docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        result = llm.invoke(
            prompt.format(
                context=context,
                question=question,
            )
        )

        memory.save_context({"input": question}, {"output": result.content})

        return result.content

    return answer_question


def clear_memory():
    """
    Reset conversation memory.
    """
    memory.clear()


if __name__ == "__main__":
    print("Paste your document:")
    text = input()

    print("Building vector store...")
    vectorstore = build_vectorstore(text)

    qa = create_qa(vectorstore)

    print("\nReady. Ask questions (type 'exit' to stop)\n")

    while True:
        q = input("Question: ")

        if q.lower() == "exit":
            break

        print("\nAnswer:", qa(q), "\n")
