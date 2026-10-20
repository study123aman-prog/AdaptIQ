"""
AdaptIQ — learn_params.py
=========================
Phase 6 — the LEARNING layer (CO5). Everything so far used *expert-set* BKT
parameters. This module makes the model learn its own parameters from logged
attempt data, which is the "forms of learning / choosing the best hypothesis"
part of the syllabus, plus a small regression/classification predictor.

Three pieces, all dependency-free plain Python:

  1. `sequence_log_likelihood` — the probability the data assigns to a set of
     answer sequences under a BKT parameter set, computed by forward filtering
     (the same two-step recursion as the tracker). This is the objective.

  2. `fit_bkt` — maximum-likelihood estimation of (p_init, p_learn, p_slip,
     p_guess) by coordinate ascent with random restarts. "Choosing the
     hypothesis (parameters) that best explains the observations."

  3. `LogisticRegression` — a from-scratch logistic-regression predictor of
     P(answer correct) from features (current mastery, question difficulty).
     Used as a readiness predictor; thresholding it is a classifier.

Plus `simulate_logs`, a BKT *generative* sampler used to produce synthetic
attempt logs for the demo and the tests. NOTE: that data is synthetic (sampled
from the model itself); with real usage logs from the app the same `fit_bkt`
would estimate parameters for real learners. The synthetic study is a
*parameter-recovery* check — fit data generated from known parameters and see
whether we get those parameters back.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from bkt import BKTParams

Seq = Sequence[bool]


# --------------------------------------------------------------------------- #
# 1. Data log-likelihood under BKT  (forward filtering)
# --------------------------------------------------------------------------- #

def sequence_log_likelihood(params: BKTParams, sequences: Sequence[Seq]) -> float:
    """
    Total log P(observed answers) under `params`, summed over all sequences.

    For each answer we (a) score it with the current predicted P(correct), then
    (b) update the belief with the two-step BKT rule — i.e. this is exactly the
    forward algorithm for the Dynamic Bayesian Network. Inlined for speed since
    the optimiser evaluates it many times.
    """
    ps, pg, pl, pinit = params.p_slip, params.p_guess, params.p_learn, params.p_init
    total = 0.0
    for seq in sequences:
        p = pinit
        for correct in seq:
            pc = p * (1.0 - ps) + (1.0 - p) * pg          # predicted P(correct)
            if correct:
                prob = pc
                num = p * (1.0 - ps)
            else:
                prob = 1.0 - pc
                num = p * ps
            total += math.log(prob if prob > 1e-12 else 1e-12)
            post = (num / prob) if prob > 1e-12 else p    # Bayes conditioning
            p = post + (1.0 - post) * pl                  # learning transition
    return total


def mean_log_likelihood(params: BKTParams, sequences: Sequence[Seq]) -> float:
    n = sum(len(s) for s in sequences)
    return sequence_log_likelihood(params, sequences) / n if n else 0.0


# --------------------------------------------------------------------------- #
# 2. Maximum-likelihood fit  (coordinate ascent + random restarts)
# --------------------------------------------------------------------------- #

# (lo, hi) search bounds for p_init, p_learn, p_slip, p_guess.
DEFAULT_BOUNDS = ((0.01, 0.95), (0.01, 0.60), (0.01, 0.40), (0.01, 0.45))


def _safe_ll(vec: List[float], sequences: Sequence[Seq]) -> float:
    try:
        return sequence_log_likelihood(BKTParams(*vec), sequences)
    except Exception:
        return float("-inf")


def _line_search(vec: List[float], i: int, sequences: Sequence[Seq],
                 bounds, points: int = 9, refines: int = 3):
    """Maximise the log-likelihood along coordinate `i`, holding the rest fixed."""
    lo, hi = bounds[i]
    if i == 2:                                   # p_slip must keep slip+guess < 1
        hi = min(hi, 0.999 - vec[3])
    elif i == 3:                                 # p_guess likewise
        hi = min(hi, 0.999 - vec[2])
    lo = max(lo, 0.001)
    if hi <= lo:
        return vec, _safe_ll(vec, sequences)

    best_x, best_ll = vec[i], _safe_ll(vec, sequences)
    win_lo, win_hi = lo, hi
    trial = list(vec)
    for _ in range(refines):
        for k in range(points):
            x = win_lo + (win_hi - win_lo) * k / (points - 1)
            trial[i] = x
            ll = _safe_ll(trial, sequences)
            if ll > best_ll:
                best_ll, best_x = ll, x
        span = (win_hi - win_lo) / 4.0
        win_lo, win_hi = max(lo, best_x - span), min(hi, best_x + span)
    out = list(vec)
    out[i] = best_x
    return out, best_ll


def _coordinate_ascent(start: List[float], sequences: Sequence[Seq],
                       bounds, sweeps: int = 30, tol: float = 1e-6):
    vec = list(start)
    ll = _safe_ll(vec, sequences)
    for _ in range(sweeps):
        improved = False
        for i in range(4):
            cand, cand_ll = _line_search(vec, i, sequences, bounds)
            if cand_ll > ll + tol:
                vec, ll, improved = cand, cand_ll, True
        if not improved:
            break
    return vec, ll


def _random_start(rng: random.Random, bounds) -> List[float]:
    return [rng.uniform(lo, hi) for (lo, hi) in bounds]


def fit_bkt(sequences: Sequence[Seq], restarts: int = 5, seed: int = 0,
            bounds=DEFAULT_BOUNDS) -> Tuple[BKTParams, float]:
    """
    Maximum-likelihood estimate of the four BKT parameters from attempt logs.
    Returns (fitted_params, log_likelihood). Uses several random restarts and
    keeps the best optimum found.
    """
    if not any(len(s) for s in sequences):
        raise ValueError("need at least one non-empty answer sequence")
    rng = random.Random(seed)
    best_vec, best_ll = None, float("-inf")
    # A couple of sensible fixed starts plus random restarts.
    starts = [[0.3, 0.1, 0.1, 0.2], [0.5, 0.2, 0.15, 0.25]]
    starts += [_random_start(rng, bounds) for _ in range(restarts)]
    for start in starts:
        vec, ll = _coordinate_ascent(start, sequences, bounds)
        if ll > best_ll:
            best_vec, best_ll = vec, ll
    return BKTParams(*best_vec), best_ll


# --------------------------------------------------------------------------- #
# 3. BKT generative sampler (synthetic attempt logs)
# --------------------------------------------------------------------------- #

def simulate_learner(params: BKTParams, n_opportunities: int,
                     rng: random.Random) -> List[bool]:
    """Sample one learner's answer sequence from the BKT generative model."""
    known = rng.random() < params.p_init
    out: List[bool] = []
    for _ in range(n_opportunities):
        if known:
            correct = rng.random() >= params.p_slip        # slip if known
        else:
            correct = rng.random() < params.p_guess        # guess if not known
        out.append(correct)
        if not known and rng.random() < params.p_learn:    # may learn before next
            known = True
    return out


def simulate_logs(params: BKTParams, n_learners: int, n_opportunities: int,
                  seed: int = 0) -> List[List[bool]]:
    rng = random.Random(seed)
    return [simulate_learner(params, n_opportunities, rng) for _ in range(n_learners)]


# --------------------------------------------------------------------------- #
# 4. Logistic-regression readiness predictor (regression + classification)
# --------------------------------------------------------------------------- #

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticRegression:
    """Binary logistic regression trained by full-batch gradient descent."""
    lr: float = 0.3
    epochs: int = 800
    l2: float = 1e-4
    w: List[float] = None
    b: float = 0.0

    def fit(self, X: List[List[float]], y: List[int]) -> "LogisticRegression":
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(X, y):
                pred = _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, xi)))
                err = pred - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * (gb / n)
        return self

    def predict_proba(self, x: List[float]) -> float:
        return _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, x)))

    def predict(self, x: List[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_proba(x) >= threshold else 0


def make_readiness_dataset(n: int, seed: int = 0):
    """
    Synthetic (mastery, difficulty) -> correct dataset from a known logistic
    ground truth, so we can show the regressor recovers the right relationship:
    higher mastery -> more likely correct; higher difficulty -> less likely.
    Difficulty is encoded 0.0 (easy) / 0.5 (medium) / 1.0 (hard).
    """
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        mastery = rng.random()
        difficulty = rng.choice([0.0, 0.5, 1.0])
        z = 3.0 * mastery - 1.8 * difficulty - 0.3
        X.append([mastery, difficulty])
        y.append(1 if rng.random() < _sigmoid(z) else 0)
    return X, y


def accuracy(model: LogisticRegression, X, y) -> float:
    return sum(1 for xi, yi in zip(X, y) if model.predict(xi) == yi) / len(y)


def log_loss(model: LogisticRegression, X, y) -> float:
    tot = 0.0
    for xi, yi in zip(X, y):
        p = min(max(model.predict_proba(xi), 1e-12), 1 - 1e-12)
        tot += -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
    return tot / len(y)


# --------------------------------------------------------------------------- #
# 5. Trivial classifier used throughout AdaptIQ
# --------------------------------------------------------------------------- #

def classify_mastery(p_mastery: float, threshold: float = 0.95) -> str:
    """Classify a topic's state from its posterior mastery probability."""
    return "mastered" if p_mastery >= threshold else "not_mastered"


# --------------------------------------------------------------------------- #
# Demo — run `python learn_params.py`
# --------------------------------------------------------------------------- #

def _demo() -> None:
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase()

    print("=" * 76)
    print("AdaptIQ — Phase 6: parameter learning")
    print("=" * 76)

    print("\nA) BKT PARAMETER RECOVERY (maximum likelihood from synthetic logs)")
    print("   Simulate learners from each topic's authored parameters, then try")
    print("   to recover those parameters by MLE.\n")
    header = f"   {'topic':<22}{'param':<9}{'true':>7}{'fitted':>9}"
    for tid in ("arrays", "recursion"):
        true = BKTParams.from_dict(kb.get_topic(tid).bkt_params)
        logs = simulate_logs(true, n_learners=600, n_opportunities=18, seed=7)
        fitted, ll = fit_bkt(logs, restarts=5, seed=1)
        print(f"   {kb.get_topic(tid).name}  (mean log-lik/answer = "
              f"{mean_log_likelihood(fitted, logs):.4f})")
        for pname in ("p_init", "p_learn", "p_slip", "p_guess"):
            print(f"       {pname:<9}{getattr(true, pname):>7.2f}{getattr(fitted, pname):>9.2f}")
        print(f"       (fitted log-lik {ll:.1f}  vs true-params log-lik "
              f"{sequence_log_likelihood(true, logs):.1f})\n")

    print("B) READINESS PREDICTOR (logistic regression: P(correct) from "
          "mastery + difficulty)")
    Xtr, ytr = make_readiness_dataset(1500, seed=1)
    Xte, yte = make_readiness_dataset(500, seed=2)
    model = LogisticRegression().fit(Xtr, ytr)
    print(f"   learned weights: mastery={model.w[0]:+.2f}, "
          f"difficulty={model.w[1]:+.2f}, bias={model.b:+.2f}")
    print(f"   test accuracy = {accuracy(model, Xte, yte):.3f}, "
          f"test log-loss = {log_loss(model, Xte, yte):.3f}")
    print("   example: P(correct | mastery=0.8, hard) = "
          f"{model.predict_proba([0.8, 1.0]):.2f}; "
          f"P(correct | mastery=0.2, easy) = {model.predict_proba([0.2, 0.0]):.2f}")


if __name__ == "__main__":
    _demo()
