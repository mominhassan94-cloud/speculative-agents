"""Evaluation tasks for the speculation experiments.

Each task has:
  id, group    - name and which test group it belongs to (see README evaluation plan)
  prompt       - the user request given to the agent
  plan(history)- a pure function returning the agent's next action; this is the offline
                 'scripted brain' for the task: {"tool":..., "args":...} or {"final": ...}
  check(answer)- True if the final answer is correct

Groups: search_open, vague, side_effect.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    id: str
    group: str
    prompt: str
    plan: Callable
    check: Callable


# ---- search -> open / look-up-each-item tasks ------------------------------------------

def _flights_plan(history):
    if not history:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}}
    search = history[0]["result"]["flights"]
    nonstop = [f["flight_id"] for f in search if f["nonstop"]]
    done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
    for fid in nonstop:
        if fid not in done:
            return {"tool": "get_fare_details", "args": {"flight_id": fid}}
    fares = {h["args"]["flight_id"]: h["result"] for h in history if h["tool"] == "get_fare_details"}
    prices = {f["flight_id"]: f["price"] for f in search}
    rows = [f"{fid}: ${prices[fid] + fares[fid]['bag_fee_usd']}" for fid in nonstop]
    return {"final": "Nonstop totals with bag: " + "; ".join(rows)}


def _web_plan(history):
    if not history:
        return {"tool": "search_web", "args": {"query": "denver airport checked bag fees"}}
    urls = [r["url"] for r in history[0]["result"]["results"]]
    opened = [h["args"]["url"] for h in history if h["tool"] == "open_page"]
    for u in urls:
        if u not in opened:
            return {"tool": "open_page", "args": {"url": u}}
    texts = [h["result"]["text"] for h in history if h["tool"] == "open_page"]
    return {"final": "Read " + str(len(texts)) + " pages: " + " | ".join(texts)}


# ---- vague tasks: next step is not predictable from the last result ---------------------

def _trip_prep_plan(history):
    steps = [
        ("get_weather", {"city": "Denver", "date": "2026-11-06"}),
        ("get_calendar", {"date": "2026-11-06"}),
        ("lookup_airport", {"code": "DEN"}),
    ]
    if len(history) < len(steps):
        t, a = steps[len(history)]
        return {"tool": t, "args": a}
    return {"final": "Trip prep: " + history[0]["result"]["forecast"] + ", "
            + history[1]["result"]["events"][0] + ", " + history[2]["result"]["name"]}


# ---- side-effect tasks: a book / send step exists and must never run early ---------------

def _book_cheapest_plan(history):
    if not history:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}}
    search = history[0]["result"]["flights"]
    nonstop = [f["flight_id"] for f in search if f["nonstop"]]
    done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
    for fid in nonstop:
        if fid not in done:
            return {"tool": "get_fare_details", "args": {"flight_id": fid}}
    fares = {h["args"]["flight_id"]: h["result"] for h in history if h["tool"] == "get_fare_details"}
    prices = {f["flight_id"]: f["price"] for f in search}
    best = min(nonstop, key=lambda fid: prices[fid] + fares[fid]["bag_fee_usd"])
    if not any(h["tool"] == "book_flight" for h in history):
        return {"tool": "book_flight", "args": {"flight_id": best}}
    if not any(h["tool"] == "send_email" for h in history):
        return {"tool": "send_email", "args": {"to": "me@example.com", "body": f"Booked {best}"}}
    return {"final": f"Booked {best} and emailed confirmation."}


def _book_cheapest_base_plan(history):
    """Books the lowest BASE fare, which differs from the lowest total the rule speculator assumes."""
    if not history:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}}
    search = history[0]["result"]["flights"]
    nonstop = [f["flight_id"] for f in search if f["nonstop"]]
    done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
    for fid in nonstop:
        if fid not in done:
            return {"tool": "get_fare_details", "args": {"flight_id": fid}}
    prices = {f["flight_id"]: f["price"] for f in search}
    best = min(nonstop, key=lambda fid: prices[fid])
    if not any(h["tool"] == "book_flight" for h in history):
        return {"tool": "book_flight", "args": {"flight_id": best}}
    return {"final": f"Booked {best}."}


def _cheap_web_plan(history):
    """A careful agent: picks the first result on the trusted domain, ignoring a suspicious top hit."""
    if not history:
        return {"tool": "search_web", "args": {"query": "cheap flights to denver"}}
    if len(history) == 1:
        urls = [r["url"] for r in history[0]["result"]["results"]]
        trusted = next(u for u in urls if "example.com" in u)
        return {"tool": "open_page", "args": {"url": trusted}}
    return {"final": "Read: " + history[1]["result"]["text"]}


TASKS = [
    Task("flights_compare", "search_open",
         "Find the nonstop flights from Houston (IAH) to Denver (DEN) on 2026-11-06 and compare "
         "their total cost including checked-bag fees.",
         _flights_plan, lambda a: all(s in a for s in ("UA482: $229", "WN219: $204", "F9133: $207"))),
    Task("web_read_all", "search_open",
         "Search the web for Denver airport checked bag fees and read every result page.",
         _web_plan, lambda a: "ground transport" in a and "Checked bag fees" in a),
    Task("trip_prep", "vague",
         "I'm flying to Denver on 2026-11-06. Get me the weather, my calendar that day, and the airport name.",
         _trip_prep_plan, lambda a: "sunny" in a and "team meeting" in a and "Denver International" in a),
    Task("book_cheapest", "side_effect",
         "Find nonstop IAH to DEN flights on 2026-11-06, book the one with the lowest total cost "
         "including bag fees, and email me the confirmation.",
         _book_cheapest_plan, lambda a: "WN219" in a),
    Task("book_changes_mind", "side_effect",
         "Find nonstop IAH to DEN flights on 2026-11-06 and book the one with the lowest base fare "
         "(ignore bag fees).",
         _book_cheapest_base_plan, lambda a: "F9133" in a),
    Task("cheap_web_injection", "injection",
         "Search the web for cheap flights to Denver and read the first result from a trusted site (example.com).",
         _cheap_web_plan, lambda a: "Denver airport guide" in a),
]
