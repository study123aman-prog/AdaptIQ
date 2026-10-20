"""
AdaptIQ — logic_engine.py
=========================
A small, from-scratch logic / inference engine (Phase 2, CO4-logic).

It provides the classic symbolic-AI machinery the syllabus asks for, with zero
third-party dependencies:

  * terms, literals and definite clauses (facts + rules);
  * **unification** with variable bindings;
  * **forward chaining** — data-driven inference to a fixpoint (handles recursive
    Datalog rules, e.g. transitive closure);
  * **backward chaining** — goal-directed proof search that returns a proof tree,
    and pinpoints the failing subgoal (this is what powers "why is X locked?");
  * **resolution refutation** for propositional logic (a standalone, gradeable
    set-piece).

The domain program itself (topics, prerequisites, mastery, eligibility rules)
lives in rules.py; this module is deliberately domain-agnostic so the inference
code can be read, tested and graded on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple, Union


# --------------------------------------------------------------------------- #
# First-order terms, literals, clauses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Var:
    """A logic variable. Constants are plain strings, so Var marks the holes."""
    name: str

    def __repr__(self) -> str:
        return f"?{self.name}"


Term = Union[str, "Var"]          # a constant (str) or a variable (Var)


@dataclass(frozen=True)
class Literal:
    """An atom like prereq(arrays, trees) or eligible(?t)."""
    pred: str
    args: Tuple[Term, ...] = ()

    def __repr__(self) -> str:
        if not self.args:
            return self.pred
        return f"{self.pred}(" + ", ".join(_term_str(a) for a in self.args) + ")"


def _term_str(t: Term) -> str:
    return repr(t) if isinstance(t, Var) else str(t)


@dataclass(frozen=True)
class Rule:
    """A definite clause: head :- body. An empty body makes it a fact."""
    head: Literal
    body: Tuple[Literal, ...] = ()

    def is_fact(self) -> bool:
        return len(self.body) == 0

    def __repr__(self) -> str:
        if self.is_fact():
            return f"{self.head}."
        return f"{self.head} :- " + ", ".join(repr(b) for b in self.body) + "."


# --------------------------------------------------------------------------- #
# Unification
# --------------------------------------------------------------------------- #

Subst = Dict[Var, Term]


def walk(t: Term, s: Subst) -> Term:
    """Follow bindings until we reach a constant or an unbound variable."""
    while isinstance(t, Var) and t in s:
        t = s[t]
    return t


def unify_term(a: Term, b: Term, s: Optional[Subst]) -> Optional[Subst]:
    if s is None:
        return None
    a, b = walk(a, s), walk(b, s)
    if a == b:
        return s
    if isinstance(a, Var):
        s2 = dict(s); s2[a] = b; return s2
    if isinstance(b, Var):
        s2 = dict(s); s2[b] = a; return s2
    return None                     # two distinct constants: clash


def unify(x: Literal, y: Literal, s: Optional[Subst] = None) -> Optional[Subst]:
    """Unify two literals, returning an extended substitution or None."""
    s = {} if s is None else dict(s)
    if x.pred != y.pred or len(x.args) != len(y.args):
        return None
    for a, b in zip(x.args, y.args):
        s = unify_term(a, b, s)
        if s is None:
            return None
    return s


def substitute(lit: Literal, s: Subst) -> Literal:
    """Apply a substitution to a literal, grounding any bound variables."""
    return Literal(lit.pred, tuple(walk(a, s) for a in lit.args))


# --------------------------------------------------------------------------- #
# Forward chaining (data-driven; computes the full closure to a fixpoint)
# --------------------------------------------------------------------------- #

def _satisfy(body: List[Literal], facts: FrozenSet[Literal],
             s: Subst) -> Iterator[Subst]:
    """Yield every substitution under which all body literals hold in `facts`."""
    if not body:
        yield s
        return
    first, rest = body[0], body[1:]
    for fact in facts:
        s2 = unify(first, fact, s)
        if s2 is not None:
            yield from _satisfy(rest, facts, s2)


def forward_chain(facts, rules: List[Rule]):
    """
    Apply the rules to the facts until no new fact can be derived.

    Returns (closure, trace) where closure is the set of all derivable facts and
    trace is the ordered list of (new_fact, rule, binding) derivation steps.
    """
    facts = set(facts)
    trace: List[Tuple[Literal, Rule, Subst]] = []
    changed = True
    while changed:
        changed = False
        snapshot = frozenset(facts)            # stable view while we scan
        derived: List[Tuple[Literal, Rule, Subst]] = []
        for rule in rules:
            for s in _satisfy(list(rule.body), snapshot, {}):
                head = substitute(rule.head, s)
                if head not in facts:
                    derived.append((head, rule, s))
        for head, rule, s in derived:
            if head not in facts:
                facts.add(head)
                trace.append((head, rule, s))
                changed = True
    return facts, trace


def ask(goal: Literal, facts, rules: List[Rule]) -> Iterator[Tuple[Subst, Literal]]:
    """
    Answer a (possibly variable-carrying) query by materialising the closure and
    matching the goal against it. Yields (binding, matched_fact).
    """
    closure, _ = forward_chain(facts, rules)
    for fact in closure:
        s = unify(goal, fact, {})
        if s is not None:
            yield s, fact


# --------------------------------------------------------------------------- #
# Backward chaining (goal-directed; builds a proof tree, finds failing subgoal)
# --------------------------------------------------------------------------- #

@dataclass
class Proof:
    """A node in a backward-chaining proof/explanation tree."""
    goal: Literal
    ok: bool
    kind: str                 # 'fact' | 'rule' | 'fail'
    children: List["Proof"]
    detail: str = ""

    def failing_leaves(self) -> List[Literal]:
        """The atomic subgoals that could not be proved (for explanations)."""
        if self.ok:
            return []
        if not self.children:
            return [self.goal]
        leaves: List[Literal] = []
        for c in self.children:
            leaves.extend(c.failing_leaves())
        return leaves or [self.goal]

    def render(self, indent: int = 0) -> str:
        mark = "✓" if self.ok else "✗"
        line = "  " * indent + f"{mark} {self.goal}"
        if self.detail:
            line += f"   [{self.detail}]"
        out = [line]
        for c in self.children:
            out.append(c.render(indent + 1))
        return "\n".join(out)


# A proper backtracking backward-chainer: it threads substitutions through the
# conjunction of subgoals (so a body-only variable — the 'M' in the transitivity
# rule — is bound correctly), standardises each rule's variables apart so
# separate uses never clash, and caps recursion depth for safety. prove()
# returns the first success as a proof tree, or an explanatory failure tree that
# names the unmet subgoal(s).

_rename_counter = 0


def _standardize_apart(rule: Rule) -> Rule:
    """Rename a rule's variables to fresh ones for this use."""
    global _rename_counter
    _rename_counter += 1
    suffix = f"#{_rename_counter}"

    def rn(t: Term) -> Term:
        return Var(t.name + suffix) if isinstance(t, Var) else t

    def rl(l: Literal) -> Literal:
        return Literal(l.pred, tuple(rn(a) for a in l.args))

    return Rule(rl(rule.head), tuple(rl(b) for b in rule.body))


def _solve_goal(goal, facts, rules, s, depth, max_depth):
    """Yield (subst, proof) for every way to prove `goal` under substitution s."""
    if depth > max_depth:
        return
    g = substitute(goal, s)
    for f in facts:                              # (1) match a fact
        s2 = unify(g, f, s)
        if s2 is not None:
            yield s2, Proof(substitute(g, s2), True, "fact", [])
    for rule in rules:                           # (2) match a head, prove its body
        r = _standardize_apart(rule)
        s2 = unify(r.head, g, s)
        if s2 is not None:
            for s3, child_proofs in _solve_conj(list(r.body), facts, rules,
                                                s2, depth + 1, max_depth):
                yield s3, Proof(substitute(g, s3), True, "rule", child_proofs)


def _solve_conj(goals, facts, rules, s, depth, max_depth):
    """Yield (subst, [proofs]) for proving a conjunction of goals left to right."""
    if not goals:
        yield s, []
        return
    first, rest = goals[0], goals[1:]
    for s1, p1 in _solve_goal(first, facts, rules, s, depth, max_depth):
        for s2, ps in _solve_conj(rest, facts, rules, s1, depth, max_depth):
            yield s2, [p1] + ps


def _provable(goal, facts, rules, max_depth: int = 100) -> bool:
    for _ in _solve_goal(goal, facts, rules, {}, 0, max_depth):
        return True
    return False


def _explain_failure(goal: Literal, facts, rules: List[Rule]) -> Proof:
    """Build a readable failure tree for a (ground) goal that cannot be proved."""
    for rule in rules:
        s = unify(rule.head, goal, {})
        if s is None:
            continue
        children: List[Proof] = []
        for b in rule.body:
            sg = substitute(b, s)
            ok = _provable(sg, facts, rules)
            children.append(Proof(sg, ok,
                                  "fact" if sg in facts else ("rule" if ok else "fail"),
                                  [], "" if ok else "not provable"))
        return Proof(goal, False, "rule", children)
    return Proof(goal, False, "fail", [], "no fact or rule proves this goal")


def prove(goal: Literal, facts, rules: List[Rule], max_depth: int = 100) -> Proof:
    """
    Backward-chain `goal`. Returns a proof tree for the first success, otherwise
    an explanatory failure tree naming the unmet subgoal(s).
    """
    for _s, proof in _solve_goal(goal, facts, rules, {}, 0, max_depth):
        return proof
    return _explain_failure(goal, facts, rules)


# --------------------------------------------------------------------------- #
# Propositional resolution refutation (standalone set-piece)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PropLit:
    """A propositional literal, e.g. mastered_recursion or ¬eligible_dp."""
    sym: str
    pos: bool = True

    def negate(self) -> "PropLit":
        return PropLit(self.sym, not self.pos)

    def __repr__(self) -> str:
        return ("" if self.pos else "¬") + self.sym


Clause = FrozenSet[PropLit]


def resolve(c1: Clause, c2: Clause) -> List[Clause]:
    """All resolvents of two clauses (one per complementary literal pair)."""
    out: List[Clause] = []
    for lit in c1:
        if lit.negate() in c2:
            out.append(frozenset((c1 - {lit}) | (c2 - {lit.negate()})))
    return out


def resolution_refutation(clauses, query: PropLit):
    """
    Decide whether KB ⊨ query by refuting KB ∪ {¬query}.

    Returns (entailed, steps) where steps is the list of (c1, c2, resolvent)
    resolutions performed (ending in the empty clause when entailed).
    """
    kb: set = {frozenset(c) for c in clauses}
    kb.add(frozenset({query.negate()}))      # assume the negation of the goal
    steps: List[Tuple[Clause, Clause, Clause]] = []

    while True:
        items = list(kb)
        new: set = set()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                for r in resolve(items[i], items[j]):
                    if len(r) == 0:
                        steps.append((items[i], items[j], r))
                        return True, steps        # empty clause ⇒ entailed
                    if r not in kb and r not in new:
                        new.add(r)
                        steps.append((items[i], items[j], r))
        if new.issubset(kb):
            return False, steps                   # no progress ⇒ not entailed
        kb |= new


# --------------------------------------------------------------------------- #
# Tiny self-test — run `python logic_engine.py`
# --------------------------------------------------------------------------- #

def _demo() -> None:
    # A toy ancestry program to exercise unification + both chaining directions.
    parent = lambda a, b: Literal("parent", (a, b))
    ancestor = lambda a, b: Literal("ancestor", (a, b))
    X, Y, Z = Var("x"), Var("y"), Var("z")

    facts = {parent("arrays", "trees"), parent("trees", "bst")}
    rules = [
        Rule(ancestor(X, Y), (parent(X, Y),)),
        Rule(ancestor(X, Z), (parent(X, Y), ancestor(Y, Z))),
    ]
    print("unify prereq pattern:",
          unify(parent(X, "bst"), parent("trees", "bst")))
    closure, trace = forward_chain(facts, rules)
    print("forward-chained ancestors:",
          sorted(repr(f) for f in closure if f.pred == "ancestor"))
    pr = prove(ancestor("arrays", "bst"), facts, rules)
    print("backward chaining ancestor(arrays, bst):", pr.ok)

    # Resolution: {a} and {¬a ∨ b} ⊨ b ?
    a, b = PropLit("a"), PropLit("b")
    entailed, steps = resolution_refutation(
        [frozenset({a}), frozenset({a.negate(), b})], b)
    print(f"resolution a, (a→b) ⊨ b : {entailed}  ({len(steps)} step/s)")


if __name__ == "__main__":
    _demo()
