import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_splitter(text: str) -> RecursiveCharacterTextSplitter:
    """
    Returns a splitter with chunk_size and overlap adapted to the document size.
    """
    length = len(text)

    if length < 5_000:  # ~2-3 pages
        chunk_size, overlap = 200, 40
    elif length < 20_000:  # ~5-15 pages
        chunk_size, overlap = 400, 80
    elif length < 50_000:  # ~15-40 pages
        chunk_size, overlap = 600, 120
    else:  # large document
        chunk_size, overlap = 1000, 200

    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)


def get_k(vectorstore) -> int:
    """
    Returns a dynamic k value based on the total number of chunks in the vectorstore.
    Bounded between 3 and 12.
    """
    total_chunks = vectorstore.index.ntotal
    return max(3, min(12, total_chunks // 5))


def build_vectorstore(text: str) -> FAISS:
    """
    Build a FAISS vectorstore from raw text.
    - Chunk size adapts to document length.
    - Each chunk carries metadata (index, total, relative position).
    """
    splitter = get_splitter(text)
    chunks = splitter.split_text(text)
    total = len(chunks)

    docs = [
        Document(
            page_content=c,
            metadata={
                "chunk_index": i,
                "chunk_total": total,
                "position": round(i / total, 2),  # 0.0 = start, 1.0 = end
            },
        )
        for i, c in enumerate(chunks)
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)


def answer_question(vectorstore: FAISS, question: str) -> str:
    """
    Stateless RAG QA function (NO MEMORY).
    - k is dynamically computed from the vectorstore size.
    """
    k = get_k(vectorstore)

    llm = ChatGroq(
        model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0
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
- If the answer is clearly absent from the context, say:
  "I cannot find this information in the provided document."
- If the answer can be reasonably inferred from the context,
  provide the inference and explicitly flag it as an inference.
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
