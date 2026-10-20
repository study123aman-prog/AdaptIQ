"""
AdaptIQ — test_learn.py
=======================
Tests for the Phase-6 learning layer. Dependency-free:

    python test_learn.py

Includes an independently hand-computed likelihood anchor (so the objective is
verified against arithmetic, not against itself), a parameter-recovery check
(fit data generated from known parameters and get them back), and logistic-
regression sanity. Fitting runs a real optimiser, so this suite takes a few
seconds.
"""

from __future__ import annotations

import math
import sys

from bkt import BKTParams, predict_correct
from learn_params import (sequence_log_likelihood, fit_bkt, simulate_logs,
                          LogisticRegression, make_readiness_dataset,
                          accuracy, classify_mastery)

P = BKTParams(p_init=0.40, p_learn=0.20, p_slip=0.10, p_guess=0.25)   # 'arrays'
_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  <-- {detail}" if (detail and not cond) else ""))
    if not cond:
        _failures.append(name)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
def test_log_likelihood_hand_values() -> None:
    """Anchor the objective against values computed by hand for P."""
    print("test_log_likelihood_hand_values")
    # One correct answer: LL = log(predict_correct(0.40)) = log(0.51).
    check("LL([correct]) == log(0.51)",
          approx(sequence_log_likelihood(P, [[True]]), math.log(0.51)),
          f"got {sequence_log_likelihood(P, [[True]])!r}")
    # One wrong answer: LL = log(1 - 0.51) = log(0.49).
    check("LL([wrong]) == log(0.49)",
          approx(sequence_log_likelihood(P, [[False]]), math.log(0.49)))
    # Two corrects: log(0.51) + log(predict_correct(update(0.40,correct))).
    # update(0.40,correct)=0.764706 -> predict=0.764706*0.9+0.235294*0.25=0.747059
    expected = math.log(0.51) + math.log(0.747059)
    check("LL([correct, correct]) matches hand value",
          approx(sequence_log_likelihood(P, [[True, True]]), expected, tol=1e-5),
          f"got {sequence_log_likelihood(P, [[True, True]])!r} want {expected!r}")


def test_likelihood_prefers_right_model() -> None:
    print("test_likelihood_prefers_right_model")
    logs = simulate_logs(P, n_learners=200, n_opportunities=16, seed=3)
    good = sequence_log_likelihood(P, logs)
    bad = sequence_log_likelihood(BKTParams(0.5, 0.3, 0.3, 0.3), logs)
    check("true params explain their own data better than a poor guess",
          good > bad, f"true={good:.1f} poor={bad:.1f}")


def test_parameter_recovery() -> None:
    print("test_parameter_recovery (fits an optimiser — a few seconds)")
    logs = simulate_logs(P, n_learners=500, n_opportunities=16, seed=7)
    poor = BKTParams(0.5, 0.3, 0.3, 0.3)
    poor_ll = sequence_log_likelihood(poor, logs)
    fitted, fit_ll = fit_bkt(logs, restarts=3, seed=1)
    true_ll = sequence_log_likelihood(P, logs)

    check("fit improves over a poor initial guess", fit_ll > poor_ll,
          f"fit={fit_ll:.1f} poor={poor_ll:.1f}")
    check("fitted log-lik is within 1% of the true-params log-lik",
          abs(fit_ll - true_ll) / abs(true_ll) < 0.01,
          f"fit={fit_ll:.1f} true={true_ll:.1f}")
    for pname in ("p_init", "p_learn", "p_slip", "p_guess"):
        tv, fv = getattr(P, pname), getattr(fitted, pname)
        check(f"recovered {pname} within 0.08 (true {tv:.2f}, got {fv:.2f})",
              abs(tv - fv) <= 0.08)


def test_logistic_regression() -> None:
    print("test_logistic_regression")
    Xtr, ytr = make_readiness_dataset(1500, seed=1)
    Xte, yte = make_readiness_dataset(500, seed=2)
    model = LogisticRegression(epochs=600).fit(Xtr, ytr)
    acc = accuracy(model, Xte, yte)
    check("test accuracy is reasonable (> 0.63)", acc > 0.63, f"acc={acc:.3f}")
    check("higher mastery raises P(correct) (positive weight)", model.w[0] > 0,
          f"w_mastery={model.w[0]:.2f}")
    check("higher difficulty lowers P(correct) (negative weight)", model.w[1] < 0,
          f"w_difficulty={model.w[1]:.2f}")
    # Monotonicity in mastery for a fixed (easy) difficulty.
    check("P(correct) increases with mastery",
          model.predict_proba([0.2, 0.0]) < model.predict_proba([0.9, 0.0]))


def test_classifier() -> None:
    print("test_classifier")
    check("0.96 -> mastered", classify_mastery(0.96) == "mastered")
    check("0.50 -> not_mastered", classify_mastery(0.50) == "not_mastered")
    check("custom threshold respected",
          classify_mastery(0.80, threshold=0.75) == "mastered")


def main() -> int:
    tests = [
        test_log_likelihood_hand_values,
        test_likelihood_prefers_right_model,
        test_parameter_recovery,
        test_logistic_regression,
        test_classifier,
    ]
    print("=" * 74)
    print("AdaptIQ learning layer — test suite")
    print("=" * 74)
    for t in tests:
        t()
    print("=" * 74)
    if _failures:
        print(f"RESULT: {len(_failures)} check(s) FAILED: {', '.join(_failures)}")
        return 1
    print("RESULT: all checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
