"""
AdaptIQ — bkt.py
================
Bayesian Knowledge Tracing (BKT) — the probabilistic student model at the heart
of AdaptIQ (Phase 1).

BKT (Corbett & Anderson, 1995) models a learner's mastery of a single skill as a
*hidden* binary state — "not yet learned" vs. "learned" — that we never observe
directly. We only see a sequence of answers (correct / incorrect) and infer
P(mastered) from them. Viewed over a sequence of questions this is a
two-node-per-step **Dynamic Bayesian Network**:

        L_t  ────────►  L_{t+1}        (hidden mastery; a Markov chain)
         │               │
         ▼               ▼
        A_t             A_{t+1}        (observed answers)

Four parameters per topic:
    p_init    P(mastered before any practice)        — the prior
    p_learn   P(not-mastered -> mastered) per step    — the temporal transition
    p_slip    P(answer wrong | mastered)              — a careless slip
    p_guess   P(answer right | not mastered)          — a lucky guess

Every observed answer triggers a TWO-STEP update of the belief p_L = P(mastered):

    1. Condition on the evidence (Bayes' theorem):
         correct :  p_L' = p_L(1-p_slip) / [ p_L(1-p_slip) + (1-p_L)p_guess ]
         wrong   :  p_L' = p_L*p_slip    / [ p_L*p_slip    + (1-p_L)(1-p_guess) ]
    2. Apply the learning transition (knowledge can only grow between steps):
         p_L'' = p_L' + (1 - p_L') * p_learn

Repeating these two steps along the answer sequence is *filtering* in the DBN —
exactly "inference in a temporal model". The one-step-ahead prediction

         P(next answer correct) = p_L(1-p_slip) + (1-p_L)p_guess

is what the planner (Phase 3) uses to choose questions and judge readiness.

Design note
-----------
Like knowledge_base.py, this module is pure, dependency-free Python: the
arithmetic is scalar, so NumPy would buy nothing. Keeping the engine transparent
makes it gradeable and lets the explainability layer narrate every number it
produces. matplotlib is used only by plot_mastery.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

_PARAM_NAMES = ("p_init", "p_learn", "p_slip", "p_guess")


class BKTError(ValueError):
    """Raised for invalid BKT parameters or unknown skills."""


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BKTParams:
    """The four Bayesian Knowledge Tracing parameters for one skill/topic."""
    p_init: float
    p_learn: float
    p_slip: float
    p_guess: float

    def __post_init__(self) -> None:
        for name in _PARAM_NAMES:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise BKTError(f"BKT parameter '{name}' must be numeric, got {v!r}.")
            if not (0.0 <= v <= 1.0):
                raise BKTError(f"BKT parameter '{name}'={v} is outside [0, 1].")
        # Identifiability guard used throughout the BKT literature: if a slip and
        # a guess could together reach/exceed certainty, a *wrong* answer would
        # paradoxically *raise* mastery and the model becomes nonsensical.
        if self.p_slip + self.p_guess >= 1.0:
            raise BKTError(
                f"p_slip + p_guess = {self.p_slip + self.p_guess:.2f} >= 1 "
                "(degenerate BKT model).")

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BKTParams":
        missing = [n for n in _PARAM_NAMES if n not in d]
        if missing:
            raise BKTError(f"Missing BKT parameter(s): {', '.join(missing)}.")
        return cls(float(d["p_init"]), float(d["p_learn"]),
                   float(d["p_slip"]), float(d["p_guess"]))


# --------------------------------------------------------------------------- #
# Pure core — these four functions ARE the whole algorithm.
# They are module-level and side-effect-free so they can be unit-tested and
# reasoned about in isolation.
# --------------------------------------------------------------------------- #

def predict_correct(p_mastery: float, params: BKTParams) -> float:
    """P(the learner answers the next question correctly) given current mastery."""
    return p_mastery * (1.0 - params.p_slip) + (1.0 - p_mastery) * params.p_guess


def condition_on_answer(p_mastery: float, correct: bool, params: BKTParams) -> float:
    """
    Step 1 — Bayes' theorem. Returns the posterior P(mastered | observed answer).
    This revises belief using the answer as evidence; it does NOT yet account for
    new learning (that is step 2).
    """
    if correct:
        num = p_mastery * (1.0 - params.p_slip)
        den = num + (1.0 - p_mastery) * params.p_guess
    else:
        num = p_mastery * params.p_slip
        den = num + (1.0 - p_mastery) * (1.0 - params.p_guess)
    if den <= 0.0:          # degenerate corner (e.g. p_mastery=0 and p_guess=0)
        return p_mastery
    return num / den


def apply_learning(p_posterior: float, params: BKTParams) -> float:
    """
    Step 2 — the temporal transition. Between two questions the learner may move
    from 'not mastered' to 'mastered' with probability p_learn; mastery never
    decreases through this step (standard BKT assumes no forgetting).
    """
    return p_posterior + (1.0 - p_posterior) * params.p_learn


def update(p_mastery: float, correct: bool, params: BKTParams) -> float:
    """The full two-step BKT update: condition on the answer, then transition."""
    return apply_learning(condition_on_answer(p_mastery, correct, params), params)


# --------------------------------------------------------------------------- #
# Reasoning trace
# --------------------------------------------------------------------------- #

@dataclass
class StepResult:
    """A fully transparent record of one observe() call — a reasoning-trace unit."""
    topic_id: str
    correct: bool
    prior: float              # P(mastered) before this answer
    predicted_correct: float  # P(correct) the model expected beforehand
    posterior: float          # after Bayes conditioning (step 1)
    mastery: float            # after the learning transition (step 2) = new belief
    step: int                 # 1-based index within this topic's history

    @property
    def delta(self) -> float:
        return self.mastery - self.prior

    def explain(self, topic_name: Optional[str] = None) -> str:
        """Human-readable narration of this single update (the glass-box view)."""
        name = topic_name or self.topic_id
        verdict = "correctly" if self.correct else "incorrectly"
        direction = "rose" if self.delta >= 0 else "fell"
        # Flag an answer the model found unlikely — it carries more information.
        p_obs = self.predicted_correct if self.correct else 1.0 - self.predicted_correct
        note = ""
        if p_obs < 0.35:
            note = (f" The model had expected this answer with probability only "
                    f"{p_obs:.2f}, so it shifted the estimate more than usual.")
        return (
            f"[{name}] You answered {verdict}. Mastery {direction} "
            f"{self.prior:.3f} -> {self.mastery:.3f} (delta {self.delta:+.3f}). "
            f"The Bayesian step revised belief to {self.posterior:.3f} using the "
            f"answer as evidence; the learning step then accounted for new "
            f"knowledge, giving {self.mastery:.3f}.{note}"
        )


# --------------------------------------------------------------------------- #
# The student model
# --------------------------------------------------------------------------- #

class MasteryTracker:
    """
    Tracks P(mastered) for every topic as a learner answers questions, using the
    two-step BKT update above. This is AdaptIQ's student model: the planner reads
    its mastery vector and predictions, and the explainer narrates its history.
    """

    def __init__(self, params: Dict[str, BKTParams]):
        if not params:
            raise BKTError("MasteryTracker needs parameters for at least one topic.")
        self._params: Dict[str, BKTParams] = dict(params)
        self._mastery: Dict[str, float] = {t: p.p_init for t, p in self._params.items()}
        self._history: Dict[str, List[StepResult]] = {t: [] for t in self._params}

    # ----------------------------------------------- construction from a KB
    @classmethod
    def from_knowledge_base(cls, kb) -> "MasteryTracker":
        """Build a tracker whose parameters come from the topic frames in a KB."""
        params = {t.id: BKTParams.from_dict(t.bkt_params) for t in kb.all_topics()}
        return cls(params)

    # ---------------------------------------------------------------- queries
    def topics(self) -> List[str]:
        return list(self._params)

    def params_of(self, topic_id: str) -> BKTParams:
        self._require(topic_id)
        return self._params[topic_id]

    def mastery_of(self, topic_id: str) -> float:
        self._require(topic_id)
        return self._mastery[topic_id]

    def probability_correct(self, topic_id: str) -> float:
        """One-step-ahead prediction P(next answer correct) for this topic."""
        self._require(topic_id)
        return predict_correct(self._mastery[topic_id], self._params[topic_id])

    def is_mastered(self, topic_id: str, threshold: float = 0.95) -> bool:
        self._require(topic_id)
        return self._mastery[topic_id] >= threshold

    def mastered_topics(self, threshold: float = 0.95) -> List[str]:
        return [t for t in self._params if self._mastery[t] >= threshold]

    def snapshot(self) -> Dict[str, float]:
        """A copy of the current mastery vector (for the planner / UI / storage)."""
        return dict(self._mastery)

    def history(self, topic_id: str) -> List[StepResult]:
        self._require(topic_id)
        return list(self._history[topic_id])

    # ----------------------------------------------------------------- update
    def observe(self, topic_id: str, correct: bool) -> StepResult:
        """Record one answer, update the belief for its topic, return the trace."""
        self._require(topic_id)
        params = self._params[topic_id]
        prior = self._mastery[topic_id]
        predicted = predict_correct(prior, params)
        posterior = condition_on_answer(prior, correct, params)
        new_mastery = apply_learning(posterior, params)
        self._mastery[topic_id] = new_mastery
        result = StepResult(
            topic_id=topic_id, correct=bool(correct), prior=prior,
            predicted_correct=predicted, posterior=posterior,
            mastery=new_mastery, step=len(self._history[topic_id]) + 1,
        )
        self._history[topic_id].append(result)
        return result

    def observe_sequence(self, topic_id: str, answers: List[bool]) -> List[StepResult]:
        return [self.observe(topic_id, a) for a in answers]

    def reset(self, topic_id: Optional[str] = None) -> None:
        """Reset one topic (or all topics) back to the prior and clear history."""
        targets = [topic_id] if topic_id is not None else list(self._params)
        for t in targets:
            self._require(t)
            self._mastery[t] = self._params[t].p_init
            self._history[t] = []

    # -------------------------------------------------------- explainability
    def explain_state(self, topic_id: str, topic_name: Optional[str] = None) -> str:
        self._require(topic_id)
        m = self._mastery[topic_id]
        pc = self.probability_correct(topic_id)
        name = topic_name or topic_id
        tag = "mastered" if self.is_mastered(topic_id) else "still learning"
        return (f"[{name}] P(mastered) = {m:.3f} ({tag}); "
                f"predicted P(correct next) = {pc:.3f}.")

    # ---------------------------------------------------------------- internal
    def _require(self, topic_id: str) -> None:
        if topic_id not in self._params:
            raise BKTError(f"Unknown topic '{topic_id}'.")


# --------------------------------------------------------------------------- #
# CLI / self-check — run `python bkt.py` to see a narrated mastery trace.
# --------------------------------------------------------------------------- #

def _demo() -> None:
    from knowledge_base import KnowledgeBase       # local import keeps core dep-free
    kb = KnowledgeBase()
    tracker = MasteryTracker.from_knowledge_base(kb)

    topic = "arrays"
    name = kb.get_topic(topic).name
    p = tracker.params_of(topic)
    print("=" * 74)
    print(f"BKT demo — tracking mastery of '{name}'")
    print(f"  params: p_init={p.p_init}, p_learn={p.p_learn}, "
          f"p_slip={p.p_slip}, p_guess={p.p_guess}")
    print("=" * 74)
    print("start:  " + tracker.explain_state(topic, name) + "\n")

    answers = [True, True, False, True, True, True]
    for ans in answers:
        print(tracker.observe(topic, ans).explain(name))

    print("\n" + tracker.explain_state(topic, name))


if __name__ == "__main__":
    _demo()
