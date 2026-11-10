"""Evaluation harness: run every task with and without speculation, several times each.

  python run_eval.py                      # all tasks, 3 reps, fast (scaled-down delays), offline
  python run_eval.py --reps 5 --scale 0.2
  python run_eval.py --tasks flights_compare,book_cheapest

Modes:
  baseline - no speculation
  rules    - RuleSpeculator (hand-written rules)
  oracle   - cheats by reading the task plan: always right, confidence 1.0. It is the upper
             bound on savings, and it deliberately guesses book/send steps to prove the
             safety gate blocks them.

Output: a CSV with one row per run (default workspace/results/eval_<timestamp>.csv)
plus a summary table printed to the terminal.
"""
import argparse
import csv
import os
import time
from collections import defaultdict

import mock_tools
from mock_tools import TOOLS
from agent import run_agent
from runtime import Runtime
from speculator import RuleSpeculator, Guess
from tasks import TASKS


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
        if not t.read_only:
            orig = t.fn

            def counted(_orig=orig, _name=t.name, **kw):
                SIDE_EFFECT_EXECUTIONS[_name] += 1
                return _orig(**kw)
            t.fn = counted


def make_speculator(mode, task):
    return {"baseline": None, "rules": RuleSpeculator(), "oracle": OracleSpeculator(task.plan)}[mode]


def run_one(task, mode, think_s):
    SIDE_EFFECT_EXECUTIONS.clear()
    rt = Runtime(TOOLS, speculator=make_speculator(mode, task))
    answer, secs, history = run_agent(task.prompt, PlanBrain(task.plan, think_s), rt, verbose=False)
    rt.pool.shutdown(wait=True)  # let in-flight speculation finish so counts are complete
    s = rt.summary()
    real_side_effects = sum(1 for h in history if not TOOLS[h["tool"]].read_only)
    executed_side_effects = sum(SIDE_EFFECT_EXECUTIONS.values())
    return {
        "task": task.id, "group": task.group, "mode": mode, "seconds": round(secs, 3),
        "correct": bool(task.check(answer or "")), "answer": answer,
        "launched": s["speculative_calls"], "hits": s["hits"], "wasted": s["wasted"],
        "blocked": sum(1 for l in rt.log if l.startswith("BLOCKED")),
        "held_back": sum(1 for l in rt.log if l.startswith("held back")),
        "est_saved_s": s["est_seconds_saved"],
        "side_effect_early": executed_side_effects > real_side_effects,
    }


def summarize(rows):
    base = defaultdict(list)
    for r in rows:
        if r["mode"] == "baseline":
            base[r["task"]].append(r["seconds"])
    base_mean = {k: sum(v) / len(v) for k, v in base.items()}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["mode"])].append(r)
    print(f"\n{'task':<18}{'mode':<10}{'time(s)':>8}{'faster':>8}{'launch':>8}{'hits':>6}"
          f"{'wasted':>8}{'blocked':>9}{'held':>6}{'correct':>9}{'early-SE':>10}")
    for (task, mode), rs in groups.items():
        n = len(rs)
        mean = lambda k: sum(r[k] for r in rs) / n
        faster = (1 - mean("seconds") / base_mean[task]) * 100 if task in base_mean else 0
        print(f"{task:<18}{mode:<10}{mean('seconds'):>8.2f}{faster:>7.0f}%{mean('launched'):>8.1f}"
              f"{mean('hits'):>6.1f}{mean('wasted'):>8.1f}{mean('blocked'):>9.1f}{mean('held_back'):>6.1f}"
              f"{sum(r['correct'] for r in rs):>6}/{n:<2}{sum(r['side_effect_early'] for r in rs):>8}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--scale", type=float, default=0.1, help="multiply tool latency by this (1.0 = realistic)")
    p.add_argument("--think", type=float, default=1.5, help="simulated LLM thinking time per step, seconds (before scaling)")
    p.add_argument("--tasks", default=None, help="comma-separated task ids (default: all)")
    p.add_argument("--modes", default="baseline,rules,oracle")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    mock_tools.LATENCY_SCALE = a.scale
    think_s = a.think * a.scale
    _instrument_side_effects()
    tasks = [t for t in TASKS if not a.tasks or t.id in a.tasks.split(",")]
    modes = a.modes.split(",")

    rows = []
    for task in tasks:
        for rep in range(a.reps):
            for mode in modes:
                r = run_one(task, mode, think_s)
                r["rep"] = rep
                rows.append(r)
                print(f"  {task.id:<18}{mode:<10}rep{rep}  {r['seconds']:.2f}s  correct={r['correct']}")

    out = a.out or os.path.join("workspace", "results", time.strftime("eval_%Y%m%d_%H%M%S.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\nwrote {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
