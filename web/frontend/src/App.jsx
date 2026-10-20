import { useEffect, useState } from "react";
import { api } from "./api";

const DIFF_BADGE = {
  easy: "bg-emerald-100 text-emerald-800",
  medium: "bg-amber-100 text-amber-800",
  hard: "bg-rose-100 text-rose-800",
};

function Pill({ className = "", children }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${className}`}>
      {children}
    </span>
  );
}

function Stat({ label, value, accent = "text-slate-800" }) {
  return (
    <div className="flex flex-col items-center rounded-lg bg-white/70 px-3 py-1.5 shadow-sm ring-1 ring-slate-200">
      <span className={`text-base font-bold tabular-nums ${accent}`}>{value}</span>
      <span className="text-[10px] uppercase tracking-wide text-slate-400">
        {label}
      </span>
    </div>
  );
}

function MasteryBars({ rows, threshold }) {
  return (
    <div className="space-y-1.5">
      {rows.map((r) => {
        const color = r.mastered
          ? "bg-emerald-500"
          : r.mastery < 0.4
          ? "bg-amber-400"
          : "bg-brand";
        return (
          <div key={r.id} className="flex items-center gap-2 text-xs">
            <div className="w-36 shrink-0 truncate text-slate-600" title={r.topic}>
              {r.topic}
            </div>
            <div className="relative h-3 flex-1 overflow-hidden rounded bg-slate-100">
              <div
                className={`h-3 rounded transition-[width] duration-700 ease-out ${color}`}
                style={{ width: `${Math.round(r.mastery * 100)}%` }}
              />
              <div
                className="absolute top-[-2px] h-4 w-px bg-slate-400"
                style={{ left: `${Math.round(threshold * 100)}%` }}
                title={`threshold ${threshold.toFixed(2)}`}
              />
            </div>
            <div className="w-9 shrink-0 text-right tabular-nums text-slate-500">
              {r.mastery.toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [topics, setTopics] = useState([]);
  const [selectedGoals, setSelectedGoals] = useState([]);
  const [threshold, setThreshold] = useState(0.95);

  const [sessionId, setSessionId] = useState(null);
  const [phase, setPhase] = useState("setup"); // setup | quiz | feedback | done
  const [item, setItem] = useState(null);
  const [why, setWhy] = useState("");
  const [snapshot, setSnapshot] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [choice, setChoice] = useState(null);
  const [history, setHistory] = useState([]); // [{correct}] for session stats

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showGraph, setShowGraph] = useState(false);
  const [showPlan, setShowPlan] = useState(false);

  useEffect(() => {
    api
      .getTopics()
      .then((t) => {
        setTopics(t);
        if (t.some((x) => x.id === "dynamic_programming")) {
          setSelectedGoals(["dynamic_programming"]);
        }
      })
      .catch((e) =>
        setError(
          `Can't reach the API (${e.message}). Is the backend running on :8000?`
        )
      );
  }, []);

  // Keyboard shortcuts: A–D / 1–4 to pick an answer, Enter to submit or advance.
  useEffect(() => {
    function onKey(e) {
      if (!sessionId || loading) return;
      // Ignore shortcuts while Ctrl/Cmd/Alt is held (so Ctrl+C / Ctrl+A don't
      // pick an answer), on key auto-repeat, and while typing in a text field.
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      const t = e.target;
      const typing =
        t &&
        (t.isContentEditable ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          (t.tagName === "INPUT" &&
            !["radio", "checkbox", "range", "button", "submit"].includes(t.type)));
      if (typing) return;
      if (phase === "quiz" && item) {
        const n = item.question.options.length;
        const letter =
          e.key.length === 1 ? "abcde".indexOf(e.key.toLowerCase()) : -1;
        if (letter >= 0 && letter < n) {
          setChoice(letter);
          return;
        }
        if (/^[1-9]$/.test(e.key)) {
          const idx = parseInt(e.key, 10) - 1;
          if (idx < n) setChoice(idx);
          return;
        }
        if (e.key === "Enter" && choice !== null) {
          e.preventDefault();
          handleSubmit();
        }
      } else if (phase === "feedback" && e.key === "Enter") {
        e.preventDefault();
        handleNext();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sessionId, phase, item, choice, loading]);

  function toggleGoal(id) {
    setSelectedGoals((g) =>
      g.includes(id) ? g.filter((x) => x !== id) : [...g, id]
    );
  }

  async function handleStart() {
    if (selectedGoals.length === 0) {
      setError("Pick at least one goal topic.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const { session_id } = await api.startSession(selectedGoals, threshold);
      setSessionId(session_id);
      setHistory([]);
      const data = await api.next(session_id);
      applyNext(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function applyNext(data) {
    setSnapshot(data.state);
    setFeedback(null);
    setChoice(null);
    if (data.complete) {
      setItem(null);
      setWhy("");
      setPhase("done");
    } else {
      setItem(data.item);
      setWhy(data.why);
      setPhase("quiz");
    }
  }

  async function handleSubmit() {
    if (choice === null) {
      setError("Select an option first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const data = await api.answer(sessionId, choice);
      setFeedback(data.feedback);
      setSnapshot(data.state);
      setHistory((h) => [...h, { correct: data.feedback.correct }]);
      setPhase("feedback");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleNext() {
    setLoading(true);
    try {
      const data = await api.next(sessionId);
      applyNext(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleRestart() {
    if (sessionId) api.endSession(sessionId).catch(() => {});
    setSessionId(null);
    setPhase("setup");
    setItem(null);
    setFeedback(null);
    setSnapshot(null);
    setChoice(null);
    setHistory([]);
    setError(null);
  }

  // --- derived session stats ---
  const answered = history.length;
  const correctCount = history.filter((h) => h.correct).length;
  const accuracy = answered ? Math.round((100 * correctCount) / answered) : 0;
  let streak = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].correct) streak++;
    else break;
  }

  const progress = snapshot?.progress;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-gradient-to-r from-white via-white to-blue-50">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <div>
            <h1 className="text-xl font-bold text-brand-dark">🧭 AdaptIQ</h1>
            <p className="text-xs text-slate-500">
              Explainable adaptive DSA tutor — Bayesian Knowledge Tracing + a
              prerequisite semantic net
            </p>
          </div>
          <div className="flex items-center gap-2">
            {sessionId && answered > 0 && (
              <div className="hidden gap-2 sm:flex">
                <Stat label="answered" value={answered} />
                <Stat
                  label="accuracy"
                  value={`${accuracy}%`}
                  accent={accuracy >= 60 ? "text-emerald-600" : "text-amber-600"}
                />
                <Stat
                  label="streak"
                  value={streak > 0 ? `🔥 ${streak}` : "0"}
                  accent="text-rose-500"
                />
              </div>
            )}
            {sessionId && (
              <button
                onClick={handleRestart}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
              >
                New session
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {error && (
          <div className="mb-4 animate-fadein rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">
            {error}
          </div>
        )}

        {!sessionId ? (
          <SetupView
            topics={topics}
            selectedGoals={selectedGoals}
            toggleGoal={toggleGoal}
            threshold={threshold}
            setThreshold={setThreshold}
            onStart={handleStart}
            loading={loading}
          />
        ) : (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
            {/* Left: quiz + reasoning */}
            <section className="space-y-4 lg:col-span-3">
              {progress && (
                <div>
                  <div className="mb-1 flex justify-between text-xs text-slate-500">
                    <span>
                      Progress: {progress.done} / {progress.total} required topics
                      mastered
                    </span>
                    <span>{Math.round(progress.fraction * 100)}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded bg-slate-200">
                    <div
                      className="h-2 rounded bg-gradient-to-r from-brand to-emerald-500 transition-[width] duration-500 ease-out"
                      style={{ width: `${Math.round(progress.fraction * 100)}%` }}
                    />
                  </div>
                </div>
              )}

              {phase === "done" && (
                <div className="animate-pop rounded-xl border border-emerald-200 bg-emerald-50 p-6 text-center">
                  <div className="animate-bounce text-4xl">🏆</div>
                  <h2 className="mt-2 text-lg font-bold text-emerald-800">
                    Goal reached!
                  </h2>
                  <p className="text-sm text-emerald-700">
                    All required topics are mastered
                    {answered > 0 ? ` — ${correctCount}/${answered} correct this run.` : "."}
                  </p>
                  <button
                    onClick={handleRestart}
                    className="mt-4 rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-dark"
                  >
                    Start another goal
                  </button>
                </div>
              )}

              {item && (phase === "quiz" || phase === "feedback") && (
                <div className="animate-fadein rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="mb-2 flex items-center gap-2">
                    <h2 className="text-base font-bold">{item.topic_name}</h2>
                    <Pill className={DIFF_BADGE[item.difficulty] || "bg-slate-100"}>
                      {item.difficulty}
                    </Pill>
                  </div>
                  <p className="mb-4 font-medium">{item.question.stem}</p>

                  <div className="space-y-2">
                    {item.question.options.map((opt, i) => {
                      const letter = String.fromCharCode(65 + i);
                      let tone =
                        "border-slate-200 hover:-translate-y-0.5 hover:border-brand hover:shadow-sm";
                      if (phase === "feedback" && feedback) {
                        if (i === feedback.correct_index)
                          tone = "border-emerald-400 bg-emerald-50";
                        else if (i === feedback.chosen_index)
                          tone = "border-rose-400 bg-rose-50";
                        else tone = "border-slate-200 opacity-60";
                      } else if (choice === i) {
                        tone = "border-brand bg-blue-50 ring-1 ring-brand";
                      }
                      return (
                        <label
                          key={i}
                          className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2 text-sm transition ${tone}`}
                        >
                          <input
                            type="radio"
                            name="option"
                            className="accent-brand"
                            disabled={phase === "feedback"}
                            checked={
                              phase === "feedback"
                                ? i === feedback.chosen_index
                                : choice === i
                            }
                            onChange={() => setChoice(i)}
                          />
                          <span className="font-semibold text-slate-500">
                            {letter}.
                          </span>
                          <span>{opt}</span>
                        </label>
                      );
                    })}
                  </div>

                  {phase === "quiz" && (
                    <div className="mt-4 flex items-center gap-3">
                      <button
                        onClick={handleSubmit}
                        disabled={loading || choice === null}
                        className="rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-dark disabled:opacity-50"
                      >
                        Submit answer
                      </button>
                      <span className="text-xs text-slate-400">
                        ⌨ keys <b>A–D</b> to pick · <b>Enter</b> to submit
                      </span>
                    </div>
                  )}

                  {phase === "feedback" && feedback && (
                    <div className="mt-4 animate-slideup space-y-3">
                      {feedback.correct ? (
                        <div className="rounded-lg bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-800">
                          ✅ Correct.
                        </div>
                      ) : (
                        <div className="rounded-lg bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-800">
                          ❌ Not quite — correct answer:{" "}
                          {String.fromCharCode(65 + feedback.correct_index)}.
                        </div>
                      )}
                      <p className="rounded-lg bg-blue-50 px-3 py-2 text-sm text-slate-700">
                        {feedback.explanation}
                      </p>
                      {feedback.misconception && (
                        <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900">
                          <strong>Common misconception:</strong>{" "}
                          {feedback.misconception}
                        </p>
                      )}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="rounded-lg border border-slate-200 p-3">
                          <div className="text-xs text-slate-500">
                            P(mastered) for this topic
                          </div>
                          <div className="text-2xl font-bold">
                            {feedback.mastery_after.toFixed(2)}
                          </div>
                          <div
                            className={
                              feedback.mastery_after >= feedback.mastery_before
                                ? "text-xs text-emerald-600"
                                : "text-xs text-rose-600"
                            }
                          >
                            {(feedback.mastery_after - feedback.mastery_before >= 0
                              ? "▲ +"
                              : "▼ ") +
                              (feedback.mastery_after -
                                feedback.mastery_before).toFixed(2)}
                          </div>
                        </div>
                        <div className="rounded-lg border border-slate-200 p-3">
                          <div className="text-xs text-slate-500">
                            Predicted P(correct) before you answered
                          </div>
                          <div className="text-2xl font-bold">
                            {feedback.predicted_correct.toFixed(2)}
                          </div>
                        </div>
                      </div>
                      <p className="text-xs italic text-slate-500">
                        🧠 {feedback.narration}
                      </p>
                      {feedback.mastered_now && (
                        <div className="animate-pop rounded-lg bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-800">
                          🎉 You've now mastered {feedback.topic_name}!
                        </div>
                      )}
                      <div className="flex items-center gap-3">
                        <button
                          onClick={handleNext}
                          disabled={loading}
                          className="rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-dark disabled:opacity-50"
                        >
                          Next question ▶
                        </button>
                        <span className="text-xs text-slate-400">
                          or press <b>Enter</b>
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {why && (
                <div className="animate-fadein rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <h3 className="mb-1 text-sm font-bold text-brand-dark">
                    ❓ Why this question? (reasoning trace)
                  </h3>
                  <pre className="whitespace-pre-wrap text-xs text-slate-600">
                    {why}
                  </pre>
                </div>
              )}
            </section>

            {/* Right: dashboard */}
            <aside className="space-y-4 lg:col-span-2">
              {snapshot && (
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <h3 className="mb-3 text-sm font-bold">📊 Mastery dashboard</h3>
                  <MasteryBars
                    rows={snapshot.mastery}
                    threshold={snapshot.threshold}
                  />
                  <p className="mt-2 text-[11px] text-slate-400">
                    Green = mastered (bar past the threshold line). Goal:{" "}
                    {snapshot.goals.join(", ")}.
                  </p>
                </div>
              )}

              {snapshot && (
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <button
                    onClick={() => setShowPlan((s) => !s)}
                    className="flex w-full items-center justify-between text-sm font-bold"
                  >
                    🗺️ Study plan (A* + partial order)
                    <span className="text-slate-400">{showPlan ? "▲" : "▼"}</span>
                  </button>
                  {showPlan && (
                    <pre className="mt-2 animate-fadein whitespace-pre-wrap text-xs text-slate-600">
                      {snapshot.plan}
                    </pre>
                  )}
                </div>
              )}

              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <button
                  onClick={() => setShowGraph((s) => !s)}
                  className="flex w-full items-center justify-between text-sm font-bold"
                >
                  🔗 Prerequisite semantic net
                  <span className="text-slate-400">{showGraph ? "▲" : "▼"}</span>
                </button>
                {showGraph && (
                  <img
                    src={api.graphUrl()}
                    alt="Prerequisite graph"
                    className="mt-2 w-full animate-fadein rounded border border-slate-100"
                  />
                )}
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}

function SetupView({
  topics,
  selectedGoals,
  toggleGoal,
  threshold,
  setThreshold,
  onStart,
  loading,
}) {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <section className="space-y-4 lg:col-span-3">
        <div className="animate-fadein rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="mb-1 text-lg font-bold">Pick what you want to master</h2>
          <p className="mb-4 text-sm text-slate-500">
            The tutor will plan a prerequisite-respecting path and quiz you
            adaptively until you reach your goal.
          </p>
          <div className="flex flex-wrap gap-2">
            {topics.map((t) => {
              const on = selectedGoals.includes(t.id);
              return (
                <button
                  key={t.id}
                  onClick={() => toggleGoal(t.id)}
                  className={`rounded-full border px-3 py-1.5 text-sm transition ${
                    on
                      ? "border-brand bg-brand text-white shadow-sm"
                      : "border-slate-300 bg-white text-slate-700 hover:-translate-y-0.5 hover:border-brand"
                  }`}
                >
                  {t.name}
                </button>
              );
            })}
          </div>

          <div className="mt-5">
            <label className="text-sm font-medium text-slate-700">
              Mastery threshold:{" "}
              <span className="tabular-nums">{threshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.8"
              max="0.99"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="mt-1 w-full accent-brand"
            />
          </div>

          <button
            onClick={onStart}
            disabled={loading || selectedGoals.length === 0}
            className="mt-5 rounded-lg bg-brand px-5 py-2 text-sm font-semibold text-white transition hover:bg-brand-dark disabled:opacity-50"
          >
            {loading ? "Starting…" : "Start session"}
          </button>
        </div>
      </section>

      <aside className="lg:col-span-2">
        <div className="animate-fadein rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="mb-2 text-sm font-bold">
            What makes this different from a quiz app?
          </h3>
          <ul className="list-disc space-y-2 pl-5 text-sm text-slate-600">
            <li>
              It keeps a <strong>probabilistic estimate of your mastery</strong>{" "}
              of each topic (Bayesian Knowledge Tracing) and updates it after
              every answer.
            </li>
            <li>
              It <strong>respects prerequisites</strong> — you won't see Dynamic
              Programming until you've shown you understand Recursion.
            </li>
            <li>
              It can <strong>explain every decision</strong> — why this question,
              and what your mastery is.
            </li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
