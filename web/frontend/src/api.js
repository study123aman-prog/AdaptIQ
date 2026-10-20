// Tiny fetch wrapper around the AdaptIQ FastAPI backend. In dev, Vite proxies
// "/api" to http://localhost:8000 (see vite.config.js).

const BASE = "/api";

async function request(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const err = await res.json();
      if (err && err.detail) detail = err.detail;
    } catch (_) {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  getTopics: () => request("GET", "/topics"),
  startSession: (goals, threshold) =>
    request("POST", "/session", { goals, threshold }),
  next: (sid) => request("GET", `/session/${sid}/next`),
  answer: (sid, choiceIndex) =>
    request("POST", `/session/${sid}/answer`, { choice_index: choiceIndex }),
  state: (sid) => request("GET", `/session/${sid}/state`),
  endSession: (sid) => request("DELETE", `/session/${sid}`),
  graphUrl: () => `${BASE}/graph`,
};
