"""
AdaptIQ — test_bkt.py
=====================
Unit tests for the Bayesian Knowledge Tracing engine. Dependency-free:

    python test_bkt.py

Exits 0 if all tests pass, 1 otherwise. Each test documents the property it
protects. The regression values in test 1 were computed BY HAND (not by the code
under test) so they genuinely lock the arithmetic.
"""

from __future__ import annotations

import sys

from bkt import (BKTParams, BKTError, predict_correct, condition_on_answer,
                 apply_learning, update, MasteryTracker, StepResult)

# Fixed parameters so the regression values are reproducible. These are exactly
# the 'arrays' topic parameters from topics.json.
P = BKTParams(p_init=0.40, p_learn=0.20, p_slip=0.10, p_guess=0.25)

_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  <-- {detail}" if (detail and not cond) else ""))
    if not cond:
        _failures.append(name)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
def test_regression_hand_computed() -> None:
    """Lock the two-step update against values worked out by hand for P."""
    print("test_regression_hand_computed")
    # Correct answer from prior 0.40:
    #   Bayes:    0.40*0.90 / (0.40*0.90 + 0.60*0.25) = 0.36/0.51 = 0.70588235
    #   Learn:    0.70588235 + (1-0.70588235)*0.20     = 0.76470588
    check("condition(correct) == 0.70588235",
          approx(condition_on_answer(0.40, True, P), 0.70588235),
          f"got {condition_on_answer(0.40, True, P)!r}")
    check("update(correct) == 0.76470588",
          approx(update(0.40, True, P), 0.76470588),
          f"got {update(0.40, True, P)!r}")
    # Wrong answer from prior 0.40  (note: 1 - p_guess = 1 - 0.25 = 0.75):
    #   Bayes:    0.40*0.10 / (0.40*0.10 + 0.60*0.75) = 0.04/0.49 = 0.08163265
    #   Learn:    0.08163265 + (1-0.08163265)*0.20     = 0.26530612
    check("condition(wrong) == 0.08163265",
          approx(condition_on_answer(0.40, False, P), 0.08163265),
          f"got {condition_on_answer(0.40, False, P)!r}")
    check("update(wrong) == 0.26530612",
          approx(update(0.40, False, P), 0.26530612),
          f"got {update(0.40, False, P)!r}")


def test_prediction_endpoints() -> None:
    """P(correct) interpolates between p_guess (ignorant) and 1-p_slip (expert)."""
    print("test_prediction_endpoints")
    check("predict(0) == p_guess", approx(predict_correct(0.0, P), P.p_guess))
    check("predict(1) == 1 - p_slip", approx(predict_correct(1.0, P), 1.0 - P.p_slip))
    check("predict(0.40) == 0.51", approx(predict_correct(0.40, P), 0.51))
    # Monotonic in mastery.
    check("predict is increasing in mastery",
          predict_correct(0.2, P) < predict_correct(0.8, P))


def test_evidence_direction() -> None:
    """A correct answer is positive evidence; a wrong answer is negative evidence.
    The Bayes-conditioning step must strictly raise belief on correct and strictly
    lower it on wrong, for any interior prior (guaranteed because p_slip+p_guess<1).
    """
    print("test_evidence_direction")
    ok_up = ok_dn = True
    for p in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        if not (condition_on_answer(p, True, P) > p):
            ok_up = False
        if not (condition_on_answer(p, False, P) < p):
            ok_dn = False
    check("correct raises the conditioned belief", ok_up)
    check("wrong lowers the conditioned belief", ok_dn)
    # And a wrong answer always leaves you lower than a correct one from same prior.
    check("update(wrong) < update(correct) from same prior",
          all(update(p, False, P) < update(p, True, P)
              for p in (0.1, 0.3, 0.5, 0.7, 0.9)))
    # For these parameters a single wrong answer produces a NET fall below prior.
    check("single wrong answer nets a fall for P", update(0.40, False, P) < 0.40)


def test_bounds_and_convergence() -> None:
    """Mastery stays in [0,1]; a correct streak -> ~1, a wrong streak -> low."""
    print("test_bounds_and_convergence")
    # All-correct: strictly increasing, crosses the 0.95 threshold.
    p = P.p_init
    seq = []
    for _ in range(6):
        p = update(p, True, P)
        seq.append(p)
    check("all-correct is monotonically increasing",
          all(b > a for a, b in zip(seq, seq[1:])))
    check("all-correct crosses 0.95 within 6 steps", seq[-1] > 0.95,
          f"ended at {seq[-1]:.4f}")
    check("mastery never exceeds 1.0", all(x <= 1.0 for x in seq))
    # All-wrong: monotonically decreasing, stays well below the prior, >= 0.
    p = P.p_init
    seqw = []
    for _ in range(6):
        p = update(p, False, P)
        seqw.append(p)
    check("all-wrong is monotonically decreasing",
          all(b < a for a, b in zip(seqw, seqw[1:])))
    check("all-wrong stays below the prior", all(x < P.p_init for x in seqw))
    check("mastery never drops below 0.0", all(x >= 0.0 for x in seqw))


def test_guess_vs_streak() -> None:
    """A single (possibly lucky) correct answer must not jump the model to
    certainty — slip/guess keep it moderate — whereas a genuine streak does
    drive mastery high. This is the hallmark BKT behaviour."""
    print("test_guess_vs_streak")
    one = update(P.p_init, True, P)
    streak = P.p_init
    for _ in range(4):
        streak = update(streak, True, P)
    check("one correct stays moderate (< 0.80)", one < 0.80, f"got {one:.4f}")
    check("a 4-correct streak is high (> 0.95)", streak > 0.95, f"got {streak:.4f}")
    check("one correct moves far less than the streak", one < streak - 0.15)


def test_param_validation() -> None:
    """Bad parameters must be rejected loudly."""
    print("test_param_validation")
    def raises(fn) -> bool:
        try:
            fn()
            return False
        except BKTError:
            return True
    check("rejects p_slip + p_guess >= 1",
          raises(lambda: BKTParams(0.3, 0.2, 0.6, 0.5)))
    check("rejects out-of-range probability",
          raises(lambda: BKTParams(0.3, 0.2, 0.1, 1.4)))
    check("rejects a boolean posing as a probability",
          raises(lambda: BKTParams(True, 0.2, 0.1, 0.2)))
    check("from_dict rejects a missing key",
          raises(lambda: BKTParams.from_dict({"p_init": 0.3, "p_learn": 0.2,
                                              "p_slip": 0.1})))
    check("from_dict accepts a complete dict",
          isinstance(BKTParams.from_dict(
              {"p_init": 0.3, "p_learn": 0.2, "p_slip": 0.1, "p_guess": 0.2}),
              BKTParams))


def test_degenerate_guard() -> None:
    """A zero-denominator corner must not crash; belief is left unchanged."""
    print("test_degenerate_guard")
    q = BKTParams(p_init=0.0, p_learn=0.0, p_slip=0.0, p_guess=0.0)
    check("condition() survives a 0/0 corner",
          approx(condition_on_answer(0.0, True, q), 0.0))


def test_tracker() -> None:
    """The MasteryTracker wrapper: state, history, prediction, reset, errors."""
    print("test_tracker")
    tr = MasteryTracker({"arrays": P})
    check("starts at the prior", approx(tr.mastery_of("arrays"), P.p_init))
    r = tr.observe("arrays", True)
    check("observe returns a StepResult", isinstance(r, StepResult))
    check("tracker matches the pure update",
          approx(tr.mastery_of("arrays"), update(P.p_init, True, P)))
    check("delta is recorded with the right sign", r.delta > 0)
    tr.observe("arrays", False)
    check("history length grows", len(tr.history("arrays")) == 2)
    check("probability_correct is in range", 0.0 <= tr.probability_correct("arrays") <= 1.0)
    tr.reset("arrays")
    check("reset restores the prior and clears history",
          approx(tr.mastery_of("arrays"), P.p_init) and len(tr.history("arrays")) == 0)
    # is_mastered threshold behaviour
    tr.observe_sequence("arrays", [True] * 8)
    check("is_mastered true after a long correct streak", tr.is_mastered("arrays"))
    # unknown topic errors
    def raises_unknown() -> bool:
        try:
            tr.mastery_of("nope")
            return False
        except BKTError:
            return True
    check("unknown topic raises BKTError", raises_unknown())


def main() -> int:
    tests = [
        test_regression_hand_computed,
        test_prediction_endpoints,
        test_evidence_direction,
        test_bounds_and_convergence,
        test_guess_vs_streak,
        test_param_validation,
        test_degenerate_guard,
        test_tracker,
    ]
    print("=" * 74)
    print("AdaptIQ BKT engine — test suite")
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
