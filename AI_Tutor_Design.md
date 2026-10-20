# AdaptIQ — An Explainable Adaptive DSA Tutor

*Design document. Project #2 (medium-scale) — serves as both a resume project and the
40-mark AI Lab course submission.*

---

## 1. The one-paragraph pitch

**AdaptIQ** is a tutoring system that teaches Data Structures & Algorithms and *adapts to
the learner*. It keeps a running, probabilistic estimate of how well you know each topic,
updates that estimate every time you answer a question (using Bayes' rule), and uses it to
decide what to teach you next — always respecting prerequisites (you won't get Dynamic
Programming questions until you've shown you understand recursion). Crucially, it can
**explain every decision**: *why* it thinks you haven't mastered a topic, *why* it chose
the next question, and *why* a topic is still locked. That explainability is the
differentiator — it turns "another quiz app" into an interpretable AI system.

**The technical heart** is *Bayesian Knowledge Tracing (BKT)* — a **Dynamic Bayesian
Network** that models your mastery of a skill as a hidden state evolving over time. This is
the same family of technique used by Duolingo, Khan Academy, and Coursera.

---

## 2. Why this project scores well (course + resume)

**For the AI course:** it maps almost 1:1 onto the syllabus. The hardest topic to include
authentically — *"probabilistic reasoning over time / inference in a temporal model"* — is
not bolted on here; it *is* the engine. See the full mapping in §4.

**For your resume:** it's a deliberate contrast to CropGuard. CropGuard is computer-vision /
deep-learning. AdaptIQ is *probabilistic sequential modeling + symbolic reasoning + planning*
— a different corner of AI. Together they say "broad AI foundation," which is exactly the
signal an intern/new-grad interviewer wants. Your one-liner becomes:

> "Built an adaptive tutoring engine that models student knowledge as a dynamic Bayesian
> network (Bayesian Knowledge Tracing) and plans personalized learning paths, with a
> reasoning trace that explains every recommendation."

Most junior candidates cannot produce a sentence that specific.

---

## 3. System architecture

Three-tier, mirroring CropGuard so you reuse skills (React + Node/Express + a Python AI
service). But here the interesting logic lives in the **Python reasoning engine**, not a
neural net.

```
  Browser — React + Tailwind SPA (web/)
      │  fetch /api/...            (Vite dev-proxies /api → Node)
      ▼
  Node / Express API (server/)     auth (JWT) · session & attempt history
      │            │
      │            └── MongoDB      users · topics · attempts · mastery snapshots
      ▼
  Python FastAPI "reasoning service" (engine/)
      ├─ Knowledge base   ontology + semantic net + frames + rules   (CO1)
      ├─ Logic engine     forward/backward chaining, unification     (CO4)
      ├─ BKT engine       Dynamic Bayesian Network mastery tracking  (CO4)
      ├─ Planner          state-space + partial-order + replanning   (CO5)
      └─ Explainer        turns every decision into a reasoning trace
```

**Why keep the three-tier split?** Same argument as CropGuard: Node owns accounts/history,
the Python service owns the AI. It also lets you *reuse the auth + history code you already
wrote*, so you spend your time on the novel AI parts, not boilerplate.

> **Scope-saver option:** if time is tight for the course deadline, you can ship the whole
> thing as **just the Python service + a Streamlit UI** (no React/Node) and add the
> full-stack layer later for the resume version. The AI content — which is what's graded —
> is identical. Decide at Phase 4.

---

## 4. Syllabus coverage map (the grading justification)

Every line item in your syllabus, and exactly where it lives in AdaptIQ. Keep this table —
it's your report's backbone and your viva cheat-sheet.

### CO1 / CO(KR) — Knowledge Representation

| Syllabus topic | Where it lives in AdaptIQ |
|---|---|
| **Rule-based system** | Pedagogy rules: `IF mastery(recursion) < 0.6 THEN block(dynamic_programming)`; `IF 3 consecutive wrong THEN drop difficulty`. A real forward-chaining rule engine. |
| **Semantic net** | The **prerequisite graph** of DSA topics (arrays → linked lists → recursion → trees → BST → heaps → graphs → DP), plus `is-a` / `part-of` links (e.g. "a BST *is-a* binary tree", "a heap *is-a* complete binary tree"). |
| **Reasoning in semantic net** | Inheritance + traversal: "can the student attempt DP?" = traverse prerequisite edges and check mastery of all ancestors. Property inheritance: a BST inherits "traversal" methods from binary-tree. |
| **Frames and slots** | Each topic is a **frame**: slots = `name, prerequisites, difficulty, concepts, common_misconceptions, question_bank, bkt_params{p_init,p_learn,p_slip,p_guess}`. Each question is a frame too. |
| **Ontological representation** | A small DSA **ontology**: classes (DataStructure, Algorithm, Technique), relations (prerequisiteOf, isA, usedIn), individuals (QuickSort isA Algorithm, usedIn Sorting). Backs the semantic net. |

### CO4 — Reasoning (logic)

| Syllabus topic | Where it lives in AdaptIQ |
|---|---|
| **Propositional logic** | Mastery facts as propositions: `mastered_recursion`, `mastered_arrays`. Rules combine them. |
| **Reasoning patterns in prop. logic** | Modus ponens in the rule engine: `mastered_recursion ∧ (mastered_recursion → eligible_trees) ⊢ eligible_trees`. |
| **First-order logic** | Generalize over students/topics: `∀t (Topic(t) ∧ ∀p (prereq(p,t) → mastered(student,p))) → eligible(student,t)`. |
| **Inference in FOL** | The eligibility/unlocking decisions are FOL inference over the knowledge base. |
| **Forward chaining** | Data-driven: from known mastered facts, derive all newly-eligible topics (what to unlock after an answer). |
| **Backward chaining** | Goal-driven: "*Why* can't I attempt DP?" → work backwards from `eligible(DP)`, find the unsatisfied prerequisite subgoal. This IS the explainability feature. |
| **Unification** | Matching rule patterns like `prereq(?p, dynamic_programming)` against facts to bind `?p`. |
| **Resolution** | Prove/refute an eligibility query by resolution refutation over the clause set (a clean, gradeable set-piece — even if the main loop uses chaining, implement resolution to demonstrate it). |

### CO4 — Uncertainty / Probabilistic Reasoning  ⭐ the core

| Syllabus topic | Where it lives in AdaptIQ |
|---|---|
| **Prior & posterior probabilities** | `P(mastered)` before a question = prior; after observing the answer = posterior. |
| **Bayes' theorem** | The BKT update *is* Bayes: `P(mastered | answer) = P(answer | mastered)P(mastered) / P(answer)`, using slip/guess as the likelihoods. |
| **Bayesian network** | BKT is a two-node-per-step Bayes net: hidden `Mastery` → observed `Answer`, with CPTs `{p_slip, p_guess}`. |
| **Probabilistic reasoning** | Choosing the next question uses expected information gain / expected mastery change. |
| **Probabilistic reasoning over time** ⭐ | Mastery is a **hidden state that evolves across timesteps** as the student learns — a Markov chain with a `p_learn` transition. |
| **Inference in temporal model** ⭐ | The forward algorithm on this **Dynamic Bayesian Network**: filter the mastery belief across the whole answer sequence. This is the single hardest syllabus topic and it's the beating heart of the app. |

### CO5 — Planning

| Syllabus topic | Where it lives in AdaptIQ |
|---|---|
| **Representation for planning** | State = per-topic mastery vector; actions = "teach/quiz topic X"; goal = all target topics mastered above threshold. |
| **Planning with state-space search** | Find a study path from current mastery to the goal (BFS/A* over mastery states, respecting prerequisites). |
| **Partial-order planning** | Independent topics (e.g. sorting vs. hashing) need no fixed order → produce a partial-order study plan, not a rigid list. |
| **Planning & acting in the real world** | Interleave planning with acting: teach, observe the answer, update beliefs, continue. |
| **Conditional planning / replanning** | Plan branches on outcomes ("if they fail trees, insert a recursion refresher"); **replan** when observed mastery diverges from the plan's assumption. |

### CO5 — Learning

| Syllabus topic | Where it lives in AdaptIQ |
|---|---|
| **Learning / forms of learning** | The system *learns the student's parameters* over time; discuss supervised vs. unsupervised vs. RL framing. |
| **Choosing the best hypothesis** | Fit BKT parameters (`p_learn, p_slip, p_guess`) to attempt data by maximum likelihood → picking the hypothesis that best explains observations. |
| **Classification** | Classify each attempt/topic state as mastered / not-mastered (threshold on posterior); or classify question difficulty. |
| **Regression** | Predict the student's probability of answering the next question correctly (a regression on mastery + difficulty) — also powers the "you're ready for the mock test" signal. |

**Takeaway for the report:** there is not a single syllabus line that lacks a concrete home.
That completeness is what earns marks on a 40-point rubric.

---

## 5. The BKT engine, explained simply (so you can defend it)

Bayesian Knowledge Tracing tracks one number per topic: **P(you've mastered it)**. Four
parameters per topic:

- **p_init** — prior probability you already knew it before starting.
- **p_learn** (transition) — chance you *learn* it between one question and the next.
- **p_slip** — chance you answer *wrong* even though you know it (careless slip).
- **p_guess** — chance you answer *right* even though you don't know it (lucky guess).

**Two steps per answer** (this is the whole algorithm):

1. **Update (Bayes) — condition on the observed answer.**
   If correct: `P(mastered|correct) = P(mastered)(1-p_slip) / [P(mastered)(1-p_slip) + (1-P(mastered))p_guess]`.
   If wrong: swap in `p_slip` and `(1-p_guess)`.
2. **Predict (transition) — account for learning over time.**
   `P(mastered)_next = P(mastered|answer) + (1 - P(mastered|answer)) · p_learn`.

Step 1 is Bayes' theorem. Step 2 is the temporal transition. Repeating them across a sequence
of answers is **filtering in a Dynamic Bayesian Network** — precisely "inference in a temporal
model." You'll implement this in ~40 lines of NumPy and it's the crown jewel of the demo.

---

## 6. Tech stack

| Tier | Technology | Notes |
|---|---|---|
| Reasoning engine | **Python** — NumPy for BKT, **plain Python (from scratch)** for the semantic net / prerequisite graph, plain Python for the rule/logic engine | zero graph-library dependency; the graph traversals (transitive prerequisites, topological order, cycle detection) and the logic engine are worth writing from scratch (gradeable + explainable, and fitting for a DSA/AI course) |
| Optional prob. lib | **pgmpy** | only if you want to show a "textbook" Bayesian network object alongside your from-scratch BKT; optional |
| API | **FastAPI + Uvicorn** | same as CropGuard — reuse what you know |
| App server | **Node/Express + JWT + Mongoose** | **reuse CropGuard's `server/` almost verbatim** — auth + history are the same shape |
| Database | **MongoDB** | users, topics, attempts, mastery snapshots |
| Frontend | **React + Vite + Tailwind** | reuse CropGuard patterns; OR Streamlit for a course-only fast path |
| Viz | a graph view of the prerequisite net with mastery heat + a mastery-over-time line chart | the two visuals that make demos land |

---

## 7. Phased build plan

Build a thin end-to-end slice first, then deepen — the same discipline that kept CropGuard
on track. Each phase is a working checkpoint.

**Phase 0 — Knowledge base (CO1).**
Define ~8–10 DSA topics as frames (JSON), the prerequisite semantic net (hand-written graph in plain Python), the small
ontology, and a starter question bank (~5 questions/topic, tagged with difficulty). Deliverable:
`knowledge_base.py` + `topics.json` + a script that draws the prerequisite graph.

**Phase 1 — BKT engine (CO4 probability). ⭐ Do this early; it's the core.**
Implement the two-step BKT update in NumPy. Unit-test it: feed a sequence of answers, assert
mastery rises with corrects and falls with wrongs, and that a lucky guess moves it less than a
genuine streak. Deliverable: `bkt.py` + tests + a mastery-over-time plot.

**Phase 2 — Logic + rule engine (CO4 logic).**
Forward chaining (unlock eligible topics), backward chaining (explain *why* something's locked),
unification, and a standalone resolution demo. Deliverable: `logic_engine.py` + `rules.py` + the
"why is DP locked?" trace.

**Phase 3 — Planner (CO5).**
State-space search from current mastery to a goal set, respecting prerequisites; partial-order
output; conditional replanning when an answer diverges from plan. Deliverable: `planner.py` +
a printed study plan that reorders when you fail a topic.

**Phase 4 — Wrap in FastAPI + decide UI.**
Endpoints: `/next-question`, `/answer` (returns updated mastery + reasoning trace), `/plan`,
`/explain`, `/mastery`. **Decision point:** React+Node (resume-grade) vs. Streamlit (course-fast).
Deliverable: running API + the UI choice.

**Phase 5 — Full-stack (resume version).**
Reuse CropGuard's `server/` (auth + history) and a new React UI: a quiz view, a live mastery
dashboard, the prerequisite graph with mastery heat, and the "Why this question?" panel.

**Phase 6 — Parameter learning (CO5 learning) + report.**
Fit BKT parameters to logged attempts by maximum likelihood; add the correctness-prediction
regressor. Write the report around §4's mapping. Deliverable: `learn_params.py` + the report +
a README with screenshots.

**Phase 7 (optional polish).** Deploy (ties to your Azure cert), or add a small cohort-analytics
view.

---

## 8. What makes it stand out (lean into these)

1. **Explainability first.** Every recommendation ships with a human-readable reason
   ("Chosen because your recursion mastery is 0.82 and trees is your weakest eligible topic at
   0.41"). Few student projects do this; it's your CropGuard-consistent theme.
2. **A real, named algorithm** (BKT / DBN) implemented from scratch — not a black box.
3. **Symbolic + probabilistic together.** You're combining logic (hard rules, prerequisites)
   with probability (fuzzy mastery). That hybrid is exactly what the AI course is about.
4. **You can dogfood it** on your own DSA prep and screenshot real sessions for the README.

---

## 9. Naming

Working title **AdaptIQ**. Alternatives: *MasterMind Tutor*, *PathWise*, *SkillTrace*,
*Cognita*. Pick before Phase 5 (it names the repo + UI).

---

*Next: when you're ready, we start at Phase 0 — I'll scaffold the knowledge base and draw the
prerequisite graph so you can see the semantic net take shape.*
