"""H8: can speculation serve a stale read? (Track F)

Hazard: read a resource, write it, read it again. A speculator that guesses the second read early (here a
simple prefetcher of the user's bookings, the kind of call pattern mining would learn) can start it BEFORE the
write. If that cached result is served after the write, the agent sees the world as it was before its own action.

Compared runtime cache policies (runtime.Runtime(cache=...)):
  step        unused guesses are dropped at every real call (current default)
  run         unused guesses wait for a match until the run ends (the pre-merge main behaviour)
  invalidate  like 'run', but a write drops cached reads in the same scope (or of unknown scope)

    python hazard.py --reps 20

Importing this module registers list_bookings; the default tool set (and so every live prompt) is unchanged.
"""
import argparse
import csv
import os
import time

import mock_tools
from mock_tools import TOOLS, WORLD, tool, _wl
from runtime import Runtime
from speculator import Guess
from tasks import Task, has
from agent import run_agent
from run_eval import PlanBrain


@tool("list_bookings", True, 0.8, "List the user's current flight and hotel bookings.", {}, scope="bookings")
def list_bookings():
    with _wl:
        return {"flights": list(WORLD["bookings"]), "hotels": list(WORLD["hotel_bookings"])}


def _book_then_confirm(history):
    if not history:
        return {"tool": "list_bookings", "args": {}}
    if len(history) == 1:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}}
    if len(history) == 2:
        best = min((f for f in history[1]["result"]["flights"] if f["nonstop"]), key=lambda f: f["price"])
        return {"tool": "book_flight", "args": {"flight_id": best["flight_id"]}}
    if len(history) == 3:
        return {"tool": "list_bookings", "args": {}}
    flights = history[3]["result"]["flights"]
    return {"final": f"Before: {history[0]['result']['flights']}. Now booked: {flights}."}


HAZARD_TASK = Task(
    "book_then_confirm", "hazard",
    "Show my bookings, book the cheapest nonstop base fare IAH to DEN on 2026-11-06, then list my bookings "
    "again to confirm it is there.",
    _book_then_confirm, has("Now booked: ['F9133']"), split="hazard",
    expect=lambda w: w["bookings"] == ["F9133"])


class BookingsPrefetcher:
    """Guesses list_bookings after every call: right at the last step, wrong (and early) before it."""
    name = "prefetch"

    def predict(self, task, history):
        if not history:
            return None
        return Guess("list_bookings", {}, 0.9)


def run(cache, think_s):
    mock_tools.reset_world()
    rt = Runtime(TOOLS, speculator=BookingsPrefetcher(), cache=cache)
    answer, secs, history = run_agent(HAZARD_TASK.prompt, PlanBrain(HAZARD_TASK.plan, think_s), rt, verbose=False)
    rt.pool.shutdown(wait=True)
    rt.close()
    s = rt.summary()
    ok = HAZARD_TASK.check(answer) and HAZARD_TASK.expect(WORLD)
    return {"cache": cache, "seconds": round(secs, 3), "correct": ok, "answer": answer,
            "hits": s["hits"], "launched": s["speculative_calls"], "stale_served": s["stale_served"],
            "invalidated": s["invalidated"], "wasted_tool_s": s["wasted_tool_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--think", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mock_tools.LATENCY_SCALE = a.scale
    rows = [run(c, a.think * a.scale) for _ in range(a.reps) for c in ("step", "run", "invalidate")]
    print(f"\n{'cache':<12}{'runs':>5}{'correct':>9}{'stale served':>14}{'hits':>6}{'invalidated':>13}{'median s':>10}")
    for c in ("step", "run", "invalidate"):
        rs = [r for r in rows if r["cache"] == c]
        secs = sorted(r["seconds"] for r in rs)
        print(f"{c:<12}{len(rs):>5}{sum(r['correct'] for r in rs):>6}/{len(rs):<2}{sum(r['stale_served'] for r in rs):>14}"
              f"{sum(r['hits'] for r in rs) / len(rs):>6.1f}{sum(r['invalidated'] for r in rs) / len(rs):>13.1f}"
              f"{secs[len(secs) // 2]:>10.3f}")
    print("example 'run' answer:", next(r["answer"] for r in rows if r["cache"] == "run"))
    out = a.out or os.path.join("workspace", "results", time.strftime("hazard_%Y%m%d_%H%M%S.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
