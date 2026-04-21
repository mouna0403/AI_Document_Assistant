import os

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def build_vectorstore(text: str):
    """
    Build FAISS index once per document.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)

    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = OllamaEmbeddings(
        model="llama3.2:3b", base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434")
    )

    return FAISS.from_documents(docs, embeddings)


def answer_question(vectorstore, question: str, k: int = 6):
    """
    Stateless RAG QA function (NO MEMORY).
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a precise document-based assistant.

RULES:
- Use ONLY the provided context.
- If the answer is not explicitly in the context, say:
  "I cannot find this information in the provided document."
- Do NOT guess or infer beyond the text.
- Be concise and factual.

CONTEXT:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    response = llm.invoke(prompt.format(context=context, question=question))

    return response.content


if __name__ == "__main__":
    text = "Example document about AI and retrieval systems."

    vs = build_vectorstore(text)

    while True:
        q = input("Question: ")
        if q == "exit":
            break

        print(answer_question(vs, q))
