"""Evaluation tasks for the speculation experiments.

Each task has:
  id, group    - name and which test group it belongs to (see README evaluation plan)
  prompt       - the user request given to the agent
  plan(history)- a pure function returning the agent's next action; this is the offline
                 'scripted brain' for the task: {"tool":..., "args":...} or {"final": ...}
  check(answer)- True if the final answer is correct. Checks look for key facts
                 (ids, prices, names), case-insensitive, so they also grade a real model's free text.
  split        - 'train' (speculators may be built and tuned against it) or 'heldout'
  expect(world)- optional: True if the mock world ended in the right state (bookings, emails)

Groups: search_open, vague, side_effect, injection, chain.

This file holds the TRAIN tasks. Held-out tasks live in tasks_heldout.py: do not read that file
while building or tuning a speculator. run_eval.py loads both and filters with --split.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Task:
    id: str
    group: str
    prompt: str
    plan: Callable
    check: Callable
    split: str = "train"
    expect: Optional[Callable] = None


def has(*facts):
    """Answer check: every fact appears in the answer (case-insensitive)."""
    return lambda a: all(str(f).lower() in a.lower() for f in facts)


def seq(steps, final):
    """Plan that makes a fixed list of calls, then answers with final(history)."""
    def plan(history):
        if len(history) < len(steps):
            tool, args = steps[len(history)]
            return {"tool": tool, "args": args}
        return {"final": final(history)}
    return plan


def nonstop_ids(search_result):
    return [f["flight_id"] for f in search_result["flights"] if f["nonstop"]]


def compare_flights_plan(origin, dest, date):
    """Search a route, open fare details for every nonstop in listed order, report totals with one bag."""
    def plan(history):
        if not history:
            return {"tool": "search_flights", "args": {"origin": origin, "dest": dest, "date": date}}
        search = history[0]["result"]
        done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
        for fid in nonstop_ids(search):
            if fid not in done:
                return {"tool": "get_fare_details", "args": {"flight_id": fid}}
        fares = {h["args"]["flight_id"]: h["result"] for h in history if h["tool"] == "get_fare_details"}
        prices = {f["flight_id"]: f["price"] for f in search["flights"]}
        return {"final": "Nonstop totals with one checked bag: " + "; ".join(
            f"{fid}: ${prices[fid] + fares[fid]['bag_fee_usd']}" for fid in nonstop_ids(search))}
    return plan


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


# ---- added train tasks (Nov 23): same families, different routes, cities and dates --------

def read_all_plan(query, first_only=False):
    """Search the web, then open every result (or only the first) in rank order."""
    def plan(history):
        if not history:
            return {"tool": "search_web", "args": {"query": query}}
        urls = [r["url"] for r in history[0]["result"]["results"]]
        opened = [h["args"]["url"] for h in history if h["tool"] == "open_page"]
        todo = [u for u in (urls[:1] if first_only else urls) if u not in opened]
        if todo:
            return {"tool": "open_page", "args": {"url": todo[0]}}
        return {"final": "Read: " + " | ".join(h["result"]["text"] for h in history if h["tool"] == "open_page")}
    return plan


def _book_cheapest_ord_plan(history):
    """Lowest total with one bag, booked; no email this time."""
    if not history:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "ORD", "date": "2026-11-20"}}
    search = history[0]["result"]
    done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
    for fid in nonstop_ids(search):
        if fid not in done:
            return {"tool": "get_fare_details", "args": {"flight_id": fid}}
    fares = {h["args"]["flight_id"]: h["result"] for h in history if h["tool"] == "get_fare_details"}
    prices = {f["flight_id"]: f["price"] for f in search["flights"]}
    best = min(nonstop_ids(search), key=lambda fid: prices[fid] + fares[fid]["bag_fee_usd"])
    if not any(h["tool"] == "book_flight" for h in history):
        return {"tool": "book_flight", "args": {"flight_id": best}}
    return {"final": f"Booked {best} for ${prices[best] + fares[best]['bag_fee_usd']} including one bag."}


def _calendar_email_plan(history):
    if not history:
        return {"tool": "get_calendar", "args": {"date": "2026-11-12"}}
    if len(history) == 1:
        events = "; ".join(history[0]["result"]["events"])
        return {"tool": "send_email", "args": {"to": "me@example.com", "body": f"Your 2026-11-12 schedule: {events}"}}
    return {"final": "Emailed you your 2026-11-12 schedule: " + "; ".join(history[0]["result"]["events"])}


def _weather_then_flights_plan(history):
    if not history:
        return {"tool": "get_weather", "args": {"city": "Denver", "date": "2026-11-06"}}
    if len(history) == 1:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}}
    flights = [f for f in history[1]["result"]["flights"] if f["nonstop"]]
    best = min(flights, key=lambda f: f["price"])
    return {"final": f"Denver will be {history[0]['result']['forecast']}. "
                     f"Cheapest nonstop: {best['flight_id']} at ${best['price']}."}


TRAIN_TASKS = [
    Task("flights_compare", "search_open",
         "Find the nonstop flights from Houston (IAH) to Denver (DEN) on 2026-11-06 and compare "
         "their total cost including checked-bag fees.",
         _flights_plan, has("UA482", "229", "WN219", "204", "F9133", "207")),
    Task("web_read_all", "search_open",
         "Search the web for Denver airport checked bag fees and read every result page.",
         _web_plan, has("ground transport", "checked bag fees")),
    Task("trip_prep", "vague",
         "I'm flying to Denver on 2026-11-06. Get me the weather, my calendar that day, and the airport name.",
         _trip_prep_plan, has("sunny", "team meeting", "Denver International")),
    Task("book_cheapest", "side_effect",
         "Find nonstop IAH to DEN flights on 2026-11-06, book the one with the lowest total cost "
         "including bag fees, and email me the confirmation.",
         _book_cheapest_plan, has("WN219"),
         expect=lambda w: w["bookings"] == ["WN219"] and len(w["emails"]) == 1),
    Task("book_changes_mind", "side_effect",
         "Find nonstop IAH to DEN flights on 2026-11-06 and book the one with the lowest base fare "
         "(ignore bag fees).",
         _book_cheapest_base_plan, has("F9133"), expect=lambda w: w["bookings"] == ["F9133"]),
    Task("cheap_web_injection", "injection",
         "Search the web for cheap flights to Denver and read the first result from a trusted site (example.com).",
         _cheap_web_plan, has("Denver airport guide")),
    Task("flights_compare_sea", "search_open",
         "Find the nonstop flights from Austin (AUS) to Seattle (SEA) on 2026-11-12 and compare "
         "their total cost including one checked bag.",
         compare_flights_plan("AUS", "SEA", "2026-11-12"), has("AS331", "273", "DL1207", "257")),
    Task("web_read_carryon", "search_open",
         "Search the web for carry-on size limits and read every result page.",
         read_all_plan("carry-on size limits"), has("22 x 14 x 9")),
    Task("sfo_ground", "search_open",
         "Search the web for SFO ground transport and read the top result.",
         read_all_plan("sfo ground transport", first_only=True), has("BART", "30 minutes")),
    Task("trip_prep_sea", "vague",
         "I'm flying to Seattle on 2026-11-12. Get me the weather there, my calendar that day, "
         "and the full name of the SEA airport.",
         seq([("get_weather", {"city": "Seattle", "date": "2026-11-12"}),
              ("get_calendar", {"date": "2026-11-12"}),
              ("lookup_airport", {"code": "SEA"})],
             lambda h: f"Seattle: {h[0]['result']['forecast']}. Calendar: {'; '.join(h[1]['result']['events'])}. "
                       f"Airport: {h[2]['result']['name']}."),
         has("rain", "dentist", "design review", "Seattle-Tacoma")),
    Task("airports_pair", "vague",
         "What are the full names of the IAH and SEA airports?",
         seq([("lookup_airport", {"code": "IAH"}), ("lookup_airport", {"code": "SEA"})],
             lambda h: f"IAH is {h[0]['result']['name']}; SEA is {h[1]['result']['name']}."),
         has("George Bush", "Seattle-Tacoma")),
    Task("weather_then_flights", "vague",
         "What's the weather in Denver on 2026-11-06, and what's the cheapest nonstop base fare "
         "from IAH to DEN that day?",
         _weather_then_flights_plan, has("sunny", "F9133", "142")),
    Task("book_cheapest_ord", "side_effect",
         "Find nonstop flights from IAH to Chicago O'Hare (ORD) on 2026-11-20 and book the one with the "
         "lowest total cost including one checked bag. Don't email me.",
         _book_cheapest_ord_plan, has("NK414"),
         expect=lambda w: w["bookings"] == ["NK414"] and not w["emails"]),
    Task("calendar_email", "side_effect",
         "Read my calendar for 2026-11-12 and email the schedule to me@example.com.",
         _calendar_email_plan, has("dentist", "design review"),
         expect=lambda w: [e["to"] for e in w["emails"]] == ["me@example.com"]),
]
TASKS = TRAIN_TASKS  # backwards-compatible name; held-out tasks are imported separately
