"""Evaluation harness: run every task with and without speculation, several times each.

  python run_eval.py                      # all tasks, 3 reps, fast (scaled-down delays), offline
  python run_eval.py --reps 5 --scale 0.2
  python run_eval.py --tasks flights_compare,book_cheapest
  python run_eval.py --split heldout --modes baseline,llm,llm+prov --reps 10      # setup 2 (live)
  python run_eval.py --brain gemini --scale 1 --modes baseline,llm,rules --reps 5  # setup 3 (live)

Modes:
  baseline - no speculation
  rules    - RuleSpeculator (hand-written rules)
  llm      - LLMSpeculator: a small Gemini model guesses (needs GEMINI_API_KEY)
  oracle   - cheats by reading the task plan: always right, confidence 1.0. It is the upper
             bound on savings, and it deliberately guesses book/send steps to prove the
             safety gate blocks them. Scripted brain only: a real agent has no plan to read.

Brains (--brain):
  scripted - the task's plan with a simulated think time (--think); offline, deterministic
  gemini   - a real Gemini agent (GEMINI_MODEL); think time is real, paths can differ per run.
             Use --scale 1 so tool latency is realistic next to real model latency.

Live timing: 'seconds' excludes the agent's rate-limit backoff (kept in seconds_raw and agent_backoff_s).
A guesser stuck in backoff simply guesses late; that cost stays in, and spec_backoff_s shows it.

Guess accuracy: every guess is scored against the agent's next real call, before any gate.
guess_name_acc = right tool; guess_exact_acc = right tool AND identical arguments (what a hit needs).

Output: a CSV with one row per run (default workspace/results/eval_<timestamp>.csv), a JSONL
per-call event log next to it, and a summary table printed to the terminal. Speed-ups are
paired per (task, rep) against baseline and shown as a geometric mean with a bootstrap 95% CI.
Mode order is shuffled within each rep so no mode systematically runs first.
"""
import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import mock_tools
from mock_tools import TOOLS
from agent import run_agent
from runtime import Runtime, _key
from speculator import RuleSpeculator, Guess
from tasks import TRAIN_TASKS
from tasks_heldout import HELDOUT_TASKS


class PlanBrain:
    """Offline agent: follows the task's plan after a simulated 'thinking' delay."""
    name = "plan"

    def __init__(self, plan, think_s):
        self.plan, self.think_s = plan, think_s

    def next_action(self, task, history):
        time.sleep(self.think_s)
        return self.plan(history)


class OracleSpeculator:
    name = "oracle"

    def __init__(self, plan):
        self.plan = plan

    def predict(self, task, history):
        a = self.plan(history)
        return Guess(a["tool"], a["args"], 1.0) if "tool" in a else None


# Count how many times each side-effect tool actually executes (to prove none run early).
SIDE_EFFECT_EXECUTIONS = defaultdict(int)


def _instrument_side_effects():
    for t in TOOLS.values():
        if not (t.read_only or t.reversible):  # reversible dry-run tools (hold_fare) are not 'real' side effects
            orig = t.fn

            def counted(_orig=orig, _name=t.name, **kw):
                SIDE_EFFECT_EXECUTIONS[_name] += 1
                return _orig(**kw)
            t.fn = counted


def make_speculator(name, task):
    if name == "baseline":
        return None
    if name == "rules":
        return RuleSpeculator()
    if name == "oracle":
        return OracleSpeculator(task.plan)
    if name == "llm":
        from speculator import LLMSpeculator
        return LLMSpeculator(TOOLS)
    raise ValueError(f"unknown speculator {name!r}")


def make_brain(kind, task, think_s):
    if kind == "scripted":
        return PlanBrain(task.plan, think_s)
    if kind == "gemini":
        from agent import GeminiBrain
        return GeminiBrain(TOOLS)
    raise ValueError(f"unknown brain {kind!r}")


def guess_accuracy(events):
    """Score each guess (made after real call k) against real call k+1: (guesses, name hits, exact hits)."""
    real = {e["step"]: e for e in events if e["kind"] == "real_call"}
    guesses = [e for e in events if e["kind"] == "guess"]
    name = exact = 0
    for g in guesses:
        nxt = real.get(g["step"] + 1)
        if nxt and nxt["tool"] == g["tool"]:
            name += 1
            if _key(nxt["tool"], nxt["args"]) == _key(g["tool"], g["args"]):
                exact += 1
    return len(guesses), name, exact


def make_runtime(mode, task):
    """Mode = speculator[+flag...]. Flags: prep (reversible prepare/rollback for writes),
    subs (substring provenance check), prov (substring + URL allowlist). Examples:
    baseline | rules | rules+prov | rules+prep | rules+prep+prov | oracle"""
    parts = mode.split("+")
    spec = make_speculator(parts[0], task)
    kw = {"allow_prepare": "prep" in parts}
    if "prov" in parts:
        kw["provenance"] = "allowlist"
    elif "subs" in parts:
        kw["provenance"] = "substring"
    return Runtime(TOOLS, speculator=spec, **kw)


def run_one(task, mode, think_s, brain="scripted"):
    if brain != "scripted" and mode.startswith("oracle"):
        raise ValueError("oracle needs the scripted brain (a real agent has no plan to read)")
    SIDE_EFFECT_EXECUTIONS.clear()
    mock_tools.reset_world()
    rt = make_runtime(mode, task)
    agent = make_brain(brain, task, think_s)
    answer, secs, history = run_agent(task.prompt, agent, rt, verbose=False)
    rt.pool.shutdown(wait=True)  # let in-flight speculation finish so counts are complete
    rt.close()                   # evict unused guesses and count their wasted tool time
    s = rt.summary()
    real_side_effects = sum(1 for h in history if not (TOOLS[h["tool"]].read_only or TOOLS[h["tool"]].reversible))
    real_read_calls = len(history) - real_side_effects
    executed_side_effects = sum(SIDE_EFFECT_EXECUTIONS.values())
    n_guess, name_hits, exact_hits = guess_accuracy(rt.events)
    spec_s = [c["spec_s"] for c in getattr(rt.speculator, "calls", [])]
    # time the agent spent waiting out HTTP 429/5xx retries is not agent or tool time: subtract it
    agent_backoff = getattr(getattr(agent, "llm", None), "backoff_s", 0.0)
    spec_backoff = getattr(getattr(rt.speculator, "llm", None), "backoff_s", 0.0)
    row = {
        "task": task.id, "group": task.group, "split": task.split, "brain": brain, "mode": mode,
        "seconds": round(secs - agent_backoff, 3), "seconds_raw": round(secs, 3),
        "agent_backoff_s": round(agent_backoff, 2), "spec_backoff_s": round(spec_backoff, 2),
        "agent_calls": len(history),
        "correct": bool(task.check(answer or "")) and (task.expect is None or bool(task.expect(mock_tools.WORLD))),
        "answer": answer,
        "launched": s["speculative_calls"], "hits": s["hits"], "wasted": s["wasted"],
        "blocked": sum(1 for l in rt.log if l.startswith("BLOCKED")),
        "held_back": sum(1 for l in rt.log if l.startswith("held back")),
        "est_saved_s": s["est_seconds_saved"],
        "wasted_tool_s": s["wasted_tool_s"],
        "speculator_calls": s["speculator_calls"],
        "precision": round(s["hits"] / s["speculative_calls"], 3) if s["speculative_calls"] else None,
        "recall": round(s["hits"] / real_read_calls, 3) if real_read_calls else None,
        "side_effect_early": executed_side_effects > real_side_effects,
        "refused": s["refused"], "prepared": s["prepared"], "prepare_hits": s["prepare_hits"],
        "holds_rolled_back": mock_tools.WORLD["holds_released"],
        "holds_left_over": len(mock_tools.WORLD["holds"]),
        "canary_fetched": mock_tools.WORLD["canary_fetches"] > 0,
        "booked": ",".join(mock_tools.WORLD["bookings"]),
        "hotel_booked": ",".join(mock_tools.WORLD["hotel_bookings"]),
        "emails_sent": len(mock_tools.WORLD["emails"]),
        "guesses": n_guess,
        "guess_name_acc": round(name_hits / n_guess, 3) if n_guess else None,
        "guess_exact_acc": round(exact_hits / n_guess, 3) if n_guess else None,
        "too_late": s["too_late"],
        "spec_s_mean": round(sum(spec_s) / len(spec_s), 3) if spec_s else None,
        "calls": json.dumps([[h["tool"], h["args"]] for h in history]),
    }
    return row, rt.events


def bootstrap_ci(xs, n=2000, seed=0):
    """95% bootstrap CI of the mean of xs."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(rows):
    """Paired against baseline per (task, rep). speedup = geomean(T_base / T_mode), 95% bootstrap CI."""
    by = defaultdict(dict)  # (task, rep) -> mode -> row
    for r in rows:
        by[(r["task"], r["rep"])][r["mode"]] = r
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["mode"])].append(r)
    hdr = (f"{'task':<18}{'mode':<17}{'median':>7}{'p95':>7}{'speedup':>9}{'95% CI':>16}"
           f"{'prec':>6}{'recall':>7}{'waste':>6}{'w-sec':>6}{'blk':>5}{'refus':>6}{'canary':>7}{'holdRB':>7}{'correct':>9}{'earlySE':>8}")
    print("\n" + hdr)
    for (task, mode), rs in groups.items():
        secs = [r["seconds"] for r in rs]
        logs = [math.log(by[(task, r["rep"])]["baseline"]["seconds"] / r["seconds"])
                for r in rs if "baseline" in by[(task, r["rep"])]]
        if logs and mode != "baseline":
            lo, hi = bootstrap_ci(logs)
            sp, ci = f"{math.exp(sum(logs) / len(logs)):.2f}x", f"[{math.exp(lo):.2f}, {math.exp(hi):.2f}]"
        else:
            sp, ci = "1.00x", ""
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        prec = [r["precision"] for r in rs if r["precision"] is not None]
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        f = lambda xs: f"{sum(xs) / len(xs):.2f}" if xs else "-"
        n = len(rs)
        print(f"{task:<18}{mode:<17}{pct(secs, .5):>7.2f}{pct(secs, .95):>7.2f}{sp:>9}{ci:>16}"
              f"{f(prec):>6}{f(rec):>7}{mean('wasted'):>6.1f}{mean('wasted_tool_s'):>6.2f}{mean('blocked'):>5.1f}{mean('refused'):>6.1f}"
              f"{sum(r['canary_fetched'] for r in rs):>5}/{n:<1}{mean('holds_rolled_back'):>7.2f}{sum(r['correct'] for r in rs):>6}/{n:<2}{sum(r['side_effect_early'] for r in rs):>6}/{n}")
    summarize_guesses(rows)


def summarize_guesses(rows):
    """Pooled over tasks: how often guesses name the right tool vs. match the whole call (H20)."""
    by_mode = defaultdict(list)
    for r in rows:
        if r["guesses"]:
            by_mode[(r["split"], r["mode"])].append(r)
    if not by_mode:
        return
    print(f"\n{'split':<9}{'mode':<17}{'guesses':>8}{'name acc':>10}{'exact acc':>11}{'too late':>10}{'spec s':>8}")
    for (split, mode), rs in sorted(by_mode.items()):
        g = sum(r["guesses"] for r in rs)
        name = sum(r["guess_name_acc"] * r["guesses"] for r in rs) / g
        exact = sum(r["guess_exact_acc"] * r["guesses"] for r in rs) / g
        spec = [r["spec_s_mean"] for r in rs if r["spec_s_mean"] is not None]
        spec_txt = f"{sum(spec) / len(spec):.2f}" if spec else "-"
        print(f"{split:<9}{mode:<17}{g:>8}{name:>10.2f}{exact:>11.2f}{sum(r['too_late'] for r in rs):>10}{spec_txt:>8}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="seed for mode-order shuffling")
    p.add_argument("--scale", type=float, default=0.1, help="multiply tool latency by this (1.0 = realistic)")
    p.add_argument("--think", type=float, default=1.5, help="simulated LLM thinking time per step, seconds (before scaling)")
    p.add_argument("--tasks", default=None, help="comma-separated task ids (default: all in --split)")
    p.add_argument("--split", default="train", choices=["train", "heldout", "all"],
                   help="which tasks to run; keep held-out unseen until a speculator is frozen")
    p.add_argument("--modes", default=None,
                   help="default: baseline,rules,rules+prov,rules+prep,rules+prep+prov,oracle (no oracle with --brain gemini)")
    p.add_argument("--brain", default="scripted", choices=["scripted", "gemini"],
                   help="who plays the agent: the task's scripted plan, or a real Gemini model")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    mock_tools.LATENCY_SCALE = a.scale
    think_s = a.think * a.scale
    _instrument_side_effects()
    pool = {"train": TRAIN_TASKS, "heldout": HELDOUT_TASKS, "all": TRAIN_TASKS + HELDOUT_TASKS}[a.split]
    tasks = [t for t in pool if not a.tasks or t.id in a.tasks.split(",")]
    modes = (a.modes or "baseline,rules,rules+prov,rules+prep,rules+prep+prov,oracle").split(",")
    if a.brain != "scripted":
        if a.modes and any(m.startswith("oracle") for m in modes):
            p.error("oracle modes need --brain scripted")
        modes = [m for m in modes if not m.startswith("oracle")]

    rng = random.Random(a.seed)
    rows, all_events = [], []
    for task in tasks:
        for rep in range(a.reps):
            order = modes[:]
            rng.shuffle(order)  # no mode always runs first/last
            for mode in order:
                try:
                    r, events = run_one(task, mode, think_s, a.brain)
                except Exception as e:  # a live API failure should not throw away the runs already done
                    print(f"  {task.id} {mode} rep{rep} FAILED: {type(e).__name__}: {e}")
                    continue
                r["rep"] = rep
                rows.append(r)
                all_events += [{"task": task.id, "mode": mode, "rep": rep, **e} for e in events]
            print(f"  {task.id:<18}rep{rep:<3} done")

    out = a.out or os.path.join("workspace", "results", time.strftime("eval_%Y%m%d_%H%M%S.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ev_path = os.path.splitext(out)[0] + ".events.jsonl"
    with open(ev_path, "w") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")
    summarize(rows)
    print(f"\nwrote {len(rows)} runs to {out} and {len(all_events)} call events to {ev_path}")


if __name__ == "__main__":
    main()
