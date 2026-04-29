from uuid import uuid4

# stockage mémoire (MVP)
sessions = {}


def create_session():
    session_id = str(uuid4())

    sessions[session_id] = {
        "text": None,
        "summary": None,
        "vectorstore": None,
        "embeddings_ready": False,
    }

    return session_id


def get_session(session_id: str):
    return sessions.get(session_id)


def reset_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
