"""Track A sweep: when does speculation pay, and does the latency model predict it? (H1, H2, break-even)

Per read step with tool latency D, agent think time T and guesser latency s, a guess starts s after the
result arrives and finishes at s + D; the agent asks for the call at T. So:
    hit:  the call is ready at max(T, s + D) instead of T + D        -> saves min(D, T - s)
    late: s >= T, the agent already moved on                          -> saves 0
    miss, unlimited backend: the wrong call runs alongside            -> costs 0 wall time (only tool-seconds)
    miss, one backend slot:  the real call waits for the wrong one    -> costs max(0, s + D - T), at most D
Expected saving per run = sum over read steps of  p * min(D, T - s)  -  (1 - p) * loss.
(The plan's draft formula min(T, D - s) swapped T and D; this sweep tests the corrected one.)

Speculator: CorruptedOracle(p), the true next call with probability p, else the same tool with wrong args.
Times are in model units (mock tool latencies are 0.7-2.0 units); --unit is wall seconds per unit.

    python sweep.py                                  # full grid, all 26 tasks (oracle needs no tuning)
    python sweep.py --quick                          # small grid, a few tasks, for a smoke test
"""
import argparse
import csv
import json
import math
import os
import time
import zlib
from collections import defaultdict
from itertools import product
from multiprocessing import Pool

import mock_tools
import run_eval
from mock_tools import TOOLS
from tasks import TRAIN_TASKS
from tasks_heldout import HELDOUT_TASKS

ALL_TASKS = {t.id: t for t in TRAIN_TASKS + HELDOUT_TASKS}


def predict(calls, T, s, p, slots):
    """Expected saving in model units for one run, from its real calls (list of [tool, args])."""
    if s >= T:
        return 0.0
    gain = loss = 0.0
    for tool, _ in calls[1:]:          # the first call is made before anything can be guessed
        t = TOOLS[tool]
        if not t.read_only:            # write tools are never speculated (no prepare in this sweep)
            continue
        gain += min(t.latency_s, T - s)
        loss += max(0.0, min(t.latency_s, s + t.latency_s - T))
    return p * gain - ((1 - p) * loss if slots == 1 else 0.0)


def _work(job):
    task_id, mode, T, s, slots, jitter, rep, unit = job
    mock_tools.LATENCY_SCALE = unit
    mock_tools.set_backend(slots, jitter, seed=zlib.crc32(f"{task_id}|{rep}".encode()))
    row, _ = run_eval.run_one(ALL_TASKS[task_id], mode, T * unit, spec_delay=s * unit, seed=rep)
    return {"task": task_id, "split": row["split"], "mode": mode, "T": T, "s": s,
            "slots": slots or 0, "jitter": jitter, "rep": rep,
            "seconds": row["seconds"] / unit, "hits": row["hits"], "launched": row["launched"],
            "too_late": row["too_late"], "wasted_tool_s": row["wasted_tool_s"] / unit,
            "correct": row["correct"], "side_effect_early": row["side_effect_early"], "calls": row["calls"]}


def build_jobs(a):
    jobs = []
    for task_id, T, slots, rep in product(a.tasks, a.T, a.slots, range(a.reps)):
        jobs.append((task_id, "baseline", T, 0.0, slots, a.jitter, rep, a.unit))
        for p, s in product(a.p, a.s):
            jobs.append((task_id, f"noisy{p:g}", T, s, slots, a.jitter, rep, a.unit))
    return jobs


def pair(rows):
    """Attach each run's paired baseline, measured saving and predicted saving."""
    base = {(r["task"], r["T"], r["slots"], r["rep"]): r["seconds"] for r in rows if r["mode"] == "baseline"}
    out = []
    for r in rows:
        if r["mode"] == "baseline":
            continue
        p = float(r["mode"][5:])
        b = base[(r["task"], r["T"], r["slots"], r["rep"])]
        out.append({**r, "p": p, "base_seconds": round(b, 4), "seconds": round(r["seconds"], 4),
                    "saved": round(b - r["seconds"], 4),
                    "predicted": round(predict(json.loads(r["calls"]), r["T"], r["s"], p, r["slots"]), 4),
                    "wasted_tool_s": round(r["wasted_tool_s"], 4)})
    return out


def report(paired):
    """Per cell (slots, s, p, T): mean measured vs predicted saving and speedup; then model fit and break-even."""
    cells = defaultdict(list)
    for r in paired:
        cells[(r["slots"], r["s"], r["p"], r["T"])].append(r)
    print(f"\n{'slots':>5}{'s':>5}{'p':>6}{'T':>6}{'measured':>10}{'predicted':>10}{'speedup':>9}{'waste/s':>9}")
    errs, preds = [], []
    for k in sorted(cells):
        rs = cells[k]
        m = sum(r["saved"] for r in rs) / len(rs)
        pr = sum(r["predicted"] for r in rs) / len(rs)
        sp = math.exp(sum(math.log(r["base_seconds"] / r["seconds"]) for r in rs) / len(rs))
        w = sum(r["wasted_tool_s"] for r in rs) / len(rs)
        errs.append(m - pr)
        preds.append(pr)
        print(f"{k[0] or 'inf':>5}{k[1]:>5g}{k[2]:>6g}{k[3]:>6g}{m:>10.2f}{pr:>10.2f}{sp:>8.2f}x{w:>9.2f}")
    mae = sum(abs(e) for e in errs) / len(errs)
    print(f"\nmodel fit over {len(cells)} cells: mean |measured - predicted| = {mae:.3f} units "
          f"(mean predicted saving {sum(preds) / len(preds):.2f}); mean bias {sum(errs) / len(errs):+.3f}")
    early = sum(r["side_effect_early"] for r in paired)
    wrong = sum(not r["correct"] for r in paired)
    print(f"safety: {early} runs with an early side effect, {wrong} wrong answers, out of {len(paired)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[0.25, 0.5, 1, 1.5, 2, 3, 4], help="think time, units")
    ap.add_argument("--p", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1], help="guess accuracy")
    ap.add_argument("--s", type=float, nargs="+", default=[0, 0.5, 1], help="guesser latency, units")
    ap.add_argument("--slots", type=int, nargs="+", default=[0, 1], help="backend slots (0 = unlimited)")
    ap.add_argument("--jitter", type=float, default=0.0, help="lognormal sigma on tool latency")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--unit", type=float, default=0.05, help="wall seconds per model unit")
    ap.add_argument("--tasks", nargs="+", default=list(ALL_TASKS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="small grid on 4 tasks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.T, a.p, a.s, a.reps = [0.5, 2], [0, 1], [0, 1], 1
        a.tasks = ["flights_compare", "web_read_all", "two_cheapest_fares", "book_cheapest"]
    a.slots = [x or None for x in a.slots]

    jobs = build_jobs(a)
    print(f"{len(jobs)} runs on {a.workers} workers (unit {a.unit}s)...")
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        rows = pool.map(_work, jobs, chunksize=4)
    print(f"done in {time.perf_counter() - t0:.0f}s")
    paired = pair(rows)

    out = a.out or os.path.join("workspace", "results", time.strftime("sweep_%Y%m%d_%H%M%S.csv"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        cols = [c for c in paired[0] if c != "calls"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(paired)
    report(paired)
    print(f"\nwrote {len(paired)} paired runs to {out}")


if __name__ == "__main__":
    main()
