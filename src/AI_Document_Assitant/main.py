from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from project_summarizer.utils.answer_question import answer_question, build_vectorstore
from project_summarizer.utils.downloader import extract_text_from_file
from project_summarizer.utils.sessions import create_session, get_session, reset_session
from project_summarizer.utils.summarizer import summarize_text

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# FRONTEND FIX (IMPORTANT)
# =========================
# ❌ NE PAS monter sur "/"
# ✔ sinon ça écrase les routes API
app.mount(
    "/static", StaticFiles(directory="src/project_summarizer/static"), name="static"
)


@app.get("/")
def home():
    return FileResponse("src/project_summarizer/static/index.html")


# =========================
# SESSION
# =========================
@app.get("/session")
def session():
    return {"session_id": create_session()}


# =========================
# UPLOAD + SUMMARY
# =========================
@app.post("/upload")
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    level: str = Form("standard"),
):

    session = get_session(session_id)

    if not session:
        return {"error": "invalid session"}

    content = await file.read()

    # extraction texte
    text = extract_text_from_file(content)

    # résumé
    summary = summarize_text(text, level)

    session["text"] = text
    session["summary"] = summary
    session["vectorstore"] = None
    session["embeddings_ready"] = False

    return {"summary": summary, "level": level}


# =========================
# START CHAT (EMBEDDINGS ONCE)
# =========================
@app.post("/start_chat")
def start_chat(session_id: str = Form(...)):

    session = get_session(session_id)

    if not session:
        return {"error": "invalid session"}

    if not session.get("embeddings_ready"):
        session["vectorstore"] = build_vectorstore(session["text"])
        session["embeddings_ready"] = True

    return {"status": "ready"}


# =========================
# ASK (RAG)
# =========================
@app.post("/ask")
def ask(session_id: str = Form(...), question: str = Form(...)):

    session = get_session(session_id)

    if not session or not session.get("vectorstore"):
        return {"error": "chat not ready"}

    answer = answer_question(session["vectorstore"], question)

    return {"answer": answer}


# =========================
# RESET
# =========================
@app.post("/reset")
def reset(session_id: str = Form(...)):
    reset_session(session_id)
    return {"status": "reset"}
