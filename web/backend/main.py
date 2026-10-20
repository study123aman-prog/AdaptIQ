"""
AdaptIQ — web/backend/main.py
=============================
FastAPI REST service that exposes the (already tested) AdaptIQ reasoning engine
to the React frontend. It is a thin layer: all the real logic lives in the
engine package (../../engine), reused here through tutor_session.TutorSession.

Run locally:
    cd web/backend
    python -m venv .venv && . .venv/bin/activate      # (Windows: .venv\\Scripts\\activate)
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then start the frontend (see web/frontend) — Vite dev-proxies /api to :8000.

Note: sessions are kept in memory, so they reset when the server restarts. That
is fine for a demo; a database would make them durable (a later upgrade).
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Dict, List, Optional

# --- make the engine package importable (it lives two levels up in /engine) ---
_ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from knowledge_base import KnowledgeBase          # noqa: E402  (engine imports)
from tutor_session import TutorSession            # noqa: E402

# --------------------------------------------------------------------------- #
app = FastAPI(title="AdaptIQ API", version="1.0.0",
              description="Explainable adaptive DSA tutor — BKT + semantic net.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared, validated knowledge base; one tutor session per browser session.
KB = KnowledgeBase()
SESSIONS: Dict[str, TutorSession] = {}
_GRAPH_PNG = os.path.join(_ENGINE, "prerequisite_graph.png")


# ------------------------------------------------------------- request models
class StartRequest(BaseModel):
    goals: List[str] = Field(..., min_length=1, description="goal topic ids")
    threshold: float = Field(0.95, ge=0.5, le=0.999)


class AnswerRequest(BaseModel):
    choice_index: int = Field(..., ge=0)


# ------------------------------------------------------------------- helpers
def _require(session_id: str) -> TutorSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session.")
    return session


def _state(session: TutorSession) -> dict:
    """The snapshot the dashboard/progress bar need after every action."""
    return {
        "complete": session.is_complete(),
        "progress": session.progress(),
        "mastery": session.mastery_rows(),
        "plan": session.plan_overview(),
        "goals": [session.name(g) for g in session.goals],
        "threshold": session.threshold,
    }


def _item_payload(item) -> Optional[dict]:
    """Serialise the served item WITHOUT the answer index (graded server-side)."""
    if item is None:
        return None
    return {
        "topic": item.topic,
        "topic_name": item.topic_name,
        "difficulty": item.difficulty,
        "target": item.target,
        "question": {
            "id": item.question.id,
            "stem": item.question.stem,
            "options": list(item.question.options),
        },
    }


# -------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "topics": len(KB.all_topics()),
            "active_sessions": len(SESSIONS)}


@app.get("/api/topics")
def topics() -> List[dict]:
    """All topics (prerequisite-first order) for the goal picker."""
    out = []
    for tid in KB.topological_order():
        t = KB.get_topic(tid)
        out.append({"id": t.id, "name": t.name,
                    "category": t.category, "difficulty": t.difficulty})
    return out


@app.post("/api/session")
def start_session(req: StartRequest) -> dict:
    for g in req.goals:
        if g not in KB.topics:
            raise HTTPException(status_code=400, detail=f"Unknown topic id '{g}'.")
    sid = uuid.uuid4().hex
    SESSIONS[sid] = TutorSession.start(req.goals, threshold=req.threshold, kb=KB)
    return {"session_id": sid}


@app.get("/api/session/{session_id}/next")
def next_question(session_id: str) -> dict:
    session = _require(session_id)
    item = session.next_item()
    return {
        "complete": item is None,
        "item": _item_payload(item),
        "why": session.why_this() if item else "",
        "state": _state(session),
    }


@app.post("/api/session/{session_id}/answer")
def answer(session_id: str, req: AnswerRequest) -> dict:
    session = _require(session_id)
    if session.current is None:
        raise HTTPException(status_code=409,
                            detail="No active question — fetch /next first.")
    opts = session.current.question.options
    if req.choice_index >= len(opts):
        raise HTTPException(status_code=400, detail="choice_index out of range.")
    feedback = session.submit(req.choice_index)
    return {"feedback": feedback, "state": _state(session)}


@app.get("/api/session/{session_id}/state")
def state(session_id: str) -> dict:
    return {"state": _state(_require(session_id))}


@app.delete("/api/session/{session_id}")
def end_session(session_id: str) -> dict:
    SESSIONS.pop(session_id, None)
    return {"ended": True}


@app.get("/api/graph")
def graph():
    if not os.path.exists(_GRAPH_PNG):
        raise HTTPException(status_code=404, detail="Graph image not generated yet.")
    return FileResponse(_GRAPH_PNG, media_type="image/png")
