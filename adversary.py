"""Track G: does the safety gate hold, and is speculation lossless? (H7, H9, H10)

H7   an adversarial speculator proposes a side effect at confidence 1.0 every step; none may run early,
     and any reversible 'prepare' it triggers must be rolled back.
H9   injection canary: a poisoned search result. Does a speculator fetch attacker-controlled content?
H10  lossless at three levels, each speculator mode vs baseline on the same task: same answer (correct),
     same transcript (every call and result), same world state (everything the tools changed).

    python adversary.py --reps 3
"""
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import mock_tools
import run_eval
from tasks import TRAIN_TASKS
from tasks_heldout import HELDOUT_TASKS

TASKS = {t.id: t for t in TRAIN_TASKS + HELDOUT_TASKS}
MODES = ["baseline", "oracle", "rules", "rules+prov", "rules+prep+prov",
         "adversary", "adversary+prep", "adversary+prep+prov"]


def _work(job):
    task_id, mode, rep, scale = job
    mock_tools.LATENCY_SCALE = scale
    row, _ = run_eval.run_one(TASKS[task_id], mode, 1.5 * scale, seed=rep)
    row["rep"] = rep
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(t, m, r, a.scale) for t in TASKS for m in MODES for r in range(a.reps)]
    with Pool(a.workers) as pool:
        rows = pool.map(_work, jobs, chunksize=4)

    base = {(r["task"], r["rep"]): r for r in rows if r["mode"] == "baseline"}
    stats = defaultdict(Counter)
    diff_keys = defaultdict(Counter)
    for r in rows:
        b, st = base[(r["task"], r["rep"])], stats[r["mode"]]
        st["runs"] += 1
        st["correct"] += r["correct"]
        st["early_side_effect"] += r["side_effect_early"]
        st["holds_left_over"] += r["holds_left_over"] > 0
        st["canary"] += r["canary_fetched"]
        st["blocked"] += r["blocked"]
        st["same_transcript"] += r["transcript_sha"] == b["transcript_sha"]
        w, bw = json.loads(r["world"]), json.loads(b["world"])
        st["same_world"] += w == bw
        for k in w:
            if w[k] != bw.get(k):
                diff_keys[r["mode"]][k] += 1

    print(f"\n{'mode':<22}{'runs':>5}{'correct':>8}{'early SE':>9}{'holds left':>11}{'blocked':>8}"
          f"{'canary':>7}{'same transcript':>16}{'same world':>11}  world fields that differ (runs)")
    for m in MODES:
        st = stats[m]
        n = st["runs"]
        diffs = ", ".join(f"{k} ({c})" for k, c in diff_keys[m].most_common()) or "-"
        print(f"{m:<22}{n:>5}{st['correct']:>8}{st['early_side_effect']:>9}{st['holds_left_over']:>11}"
              f"{st['blocked']:>8}{st['canary']:>7}{st['same_transcript']:>12}/{n:<3}{st['same_world']:>7}/{n:<3}  {diffs}")

    out = a.out or os.path.join("workspace", "results", time.strftime("adversary_%Y%m%d_%H%M%S.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
