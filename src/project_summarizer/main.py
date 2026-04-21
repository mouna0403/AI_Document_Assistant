import os

import streamlit as st

from project_summarizer.utils.answer_question import answer_question, build_vectorstore
from project_summarizer.utils.downloader import extract_text_from_file
from project_summarizer.utils.summarizer import summarize_text

st.set_page_config(page_title="Document Q&A", layout="centered")

st.title("📄 Document Summarizer & Q&A")

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY")
    st.stop()


# ---------------- STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "text" not in st.session_state:
    st.session_state.text = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summary_level" not in st.session_state:
    st.session_state.summary_level = None

if "qa_enabled" not in st.session_state:
    st.session_state.qa_enabled = False


# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload document", type=["pdf", "docx", "txt", "csv"]
)

summary_level = st.radio("🧠 Summary level", ["brief", "standard", "detailed"], index=1)


if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    st.session_state.text = text

    if text:
        st.success("✔ Document loaded")

        # ---------------- SUMMARY CACHE ----------------
        if (
            st.session_state.summary is None
            or st.session_state.summary_level != summary_level
        ):
            st.session_state.summary = summarize_text(text, level=summary_level)
            st.session_state.summary_level = summary_level

        st.subheader("📝 Summary")
        st.write(st.session_state.summary)

        # ---------------- Q&A ACTIVATION ----------------
        if not st.session_state.qa_enabled:
            choice = st.radio(
                "💬 Do you want to ask questions about this document?", ["No", "Yes"]
            )

            if choice == "Yes":
                with st.spinner("🔍 Preparing your document..."):
                    st.session_state.vectorstore = build_vectorstore(text)
                    st.session_state.qa_enabled = True

                st.success("✅ Ready for questions")


# ---------------- Q&A ----------------
if st.session_state.qa_enabled and st.session_state.vectorstore is not None:

    st.subheader("❓ Ask your questions")

    question = st.text_input("Type your question")

    if question:
        with st.spinner("🤖 Thinking..."):
            answer = answer_question(st.session_state.vectorstore, question)
            st.write(answer)


# ---------------- RESET ----------------
if st.button("🔄 Reset session"):
    st.session_state.clear()
    st.rerun()
