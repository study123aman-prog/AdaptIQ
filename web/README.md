# AdaptIQ — Full Web App (React + FastAPI)

A two-tier web version of AdaptIQ. The hard part — the reasoning engine
(Bayesian Knowledge Tracing, the prerequisite semantic net, the logic engine,
the planner, parameter learning) — lives in `../engine` and is fully unit-tested.
This folder only *packages* it as a website.

```
  Browser (React + Vite + Tailwind, :5173)
      │   fetch("/api/...")        ← Vite dev-proxies /api → :8000
      ▼
  FastAPI service (:8000)          ← thin REST layer over the engine
      │   imports
      ▼
  ../engine  (knowledge_base, bkt, logic_engine, planner, tutor_session, …)
```

The React app reuses the exact logic the Streamlit prototype used (via
`tutor_session.TutorSession`), so behaviour is identical; this version just has a
custom, deployable UI.

---

## Prerequisites

- **Python 3.10+** (for the backend)
- **Node 18+** and npm (for the frontend)

> Note: these two tiers could not be run inside the build sandbox (its proxy
> blocks installing FastAPI / React), so they were written and statically
> checked but must be run on your machine. If anything misbehaves on first run,
> the error messages from `uvicorn` / `vite` are specific — send them over.

---

## 1. Backend (FastAPI)

```bash
cd web/backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it: open http://localhost:8000/docs (interactive Swagger UI) or
http://localhost:8000/api/health .

The graph panel serves `../engine/prerequisite_graph.png`. It already exists; if
you ever need to regenerate it: `cd engine && python draw_graph.py`.

## 2. Frontend (React + Vite)

In a second terminal:

```bash
cd web/frontend
npm install
npm run dev
```

Open the URL Vite prints (default http://localhost:5173). The dev server proxies
`/api` to the backend on :8000, so no CORS setup is needed in development.

---

## Reopening it later

**One click (Windows):** double-click **`start_adaptiq.bat`** in the project root
(`Project_3\`). It opens the backend and the frontend in two terminals and then
launches your browser. Close those two windows to stop the app.

**Manually:** run the commands from sections 1 and 2 again in two terminals. After
the first time you can skip the installs — just `uvicorn main:app --reload --port
8000` in `web/backend` (activate the venv first) and `npm run dev` in
`web/frontend`.

**Keyboard shortcuts (during a quiz):** press **A–D** (or **1–4**) to choose an
answer, **Enter** to submit, then **Enter** again to go to the next question.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/topics` | all topics (for the goal picker) |
| POST | `/api/session` | start a session `{goals, threshold}` → `{session_id}` |
| GET  | `/api/session/{id}/next` | next adaptively-chosen question (+ reasoning + state) |
| POST | `/api/session/{id}/answer` | grade `{choice_index}` → feedback + updated state |
| GET  | `/api/session/{id}/state` | current mastery / progress / plan |
| GET  | `/api/graph` | the prerequisite-graph PNG |

Question answer keys are **never** sent to the browser — grading happens
server-side, which is both more correct and more secure.

---

## Deploying (optional, for a live resume link)

- Backend: any Python host (Render, Railway, Fly.io). Start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT`; make sure the `engine/` folder
  ships alongside `web/backend/`.
- Frontend: `npm run build` → deploy the static `dist/` to Vercel/Netlify, and
  point `/api` at your deployed backend URL (set a proxy/rewrite or an env-based
  base URL).

## Notes / future upgrades

- Sessions are stored **in memory**, so they reset when the backend restarts —
  fine for a demo. A database (or Redis) would make them durable and is the
  natural next step if you want saved history / accounts.
- Styling is Tailwind v3; the mastery dashboard uses lightweight CSS bars (no
  charting dependency) to keep the build small and robust.
