"""HELD-OUT evaluation tasks. Do not read this file while building or tuning a speculator.

These are scored only after a speculator is frozen (PLAN.md, Phase 1, track B). They are written
so that hand-written rules and patterns mined from the train tasks cannot simply match them:
  - results are opened in price order, not listed order
  - the right web result is not the first one
  - hotels: a search -> details -> book family that never appears in train
  - chains that start with an airport lookup, or run two searches back to back
  - a booking that is not preceded by fare lookups
Some reuse train patterns on purpose, so the split also measures what does transfer.
"""
from tasks import Task, has, seq, nonstop_ids, compare_flights_plan


def _city_to_flights_plan(history):
    if not history:
        return {"tool": "find_airports", "args": {"city": "Austin"}}
    if len(history) == 1:
        return {"tool": "find_airports", "args": {"city": "Seattle"}}
    if len(history) == 2:
        return {"tool": "search_flights", "args": {"origin": history[0]["result"]["airports"][0]["code"],
                                                    "dest": history[1]["result"]["airports"][0]["code"],
                                                    "date": "2026-11-12"}}
    best = min((f for f in history[2]["result"]["flights"] if f["nonstop"]), key=lambda f: f["price"])
    return {"final": f"Cheapest nonstop: {best['flight_id']} ({best['airline']}) at ${best['price']}."}


def _two_cheapest_fares_plan(history):
    """Fare details for the two cheapest nonstops, cheapest first (not listed order)."""
    if not history:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "SFO", "date": "2026-11-12"}}
    flights = history[0]["result"]["flights"]
    price = {f["flight_id"]: f["price"] for f in flights}
    two = sorted(nonstop_ids(history[0]["result"]), key=price.get)[:2]
    done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
    for fid in two:
        if fid not in done:
            return {"tool": "get_fare_details", "args": {"flight_id": fid}}
    bag = {h["args"]["flight_id"]: h["result"]["bag_fee_usd"] for h in history if h["tool"] == "get_fare_details"}
    totals = {fid: price[fid] + bag[fid] for fid in two}
    best = min(two, key=totals.get)
    return {"final": "With one bag: " + "; ".join(f"{fid}: ${totals[fid]}" for fid in two) + f". {best} is cheaper."}


def _hotels_compare_plan(history):
    if not history:
        return {"tool": "search_hotels", "args": {"city": "Seattle", "date": "2026-11-12"}}
    hotels = [h for h in history[0]["result"]["hotels"] if h["price_per_night"] < 200]
    done = [h["args"]["hotel_id"] for h in history if h["tool"] == "get_hotel_details"]
    for h in hotels:
        if h["hotel_id"] not in done:
            return {"tool": "get_hotel_details", "args": {"hotel_id": h["hotel_id"]}}
    rating = {h["args"]["hotel_id"]: h["result"]["rating"] for h in history if h["tool"] == "get_hotel_details"}
    best = max(hotels, key=lambda h: rating[h["hotel_id"]])
    return {"final": f"Best rated under $200: {best['name']} ({rating[best['hotel_id']]} stars, "
                     f"${best['price_per_night']}/night)."}


def _trusted_result_plan(query, must_contain=""):
    """Search, then open the first result hosted on example.com (optionally whose URL contains a word)."""
    def plan(history):
        if not history:
            return {"tool": "search_web", "args": {"query": query}}
        if len(history) == 1:
            urls = [r["url"] for r in history[0]["result"]["results"]]
            url = next(u for u in urls if u.startswith("https://example.com/") and must_contain in u)
            return {"tool": "open_page", "args": {"url": url}}
        return {"final": "Read: " + history[1]["result"]["text"]}
    return plan


def _book_hotel_plan(history):
    if not history:
        return {"tool": "search_hotels", "args": {"city": "Chicago", "date": "2026-11-20"}}
    cheap = [h for h in history[0]["result"]["hotels"] if h["price_per_night"] < 160]
    details = {h["args"]["hotel_id"]: h["result"] for h in history if h["tool"] == "get_hotel_details"}
    for h in cheap:
        if h["hotel_id"] not in details:
            return {"tool": "get_hotel_details", "args": {"hotel_id": h["hotel_id"]}}
    ok = [h for h in cheap if details[h["hotel_id"]]["free_cancellation"]]
    best = max(ok, key=lambda h: details[h["hotel_id"]]["rating"])
    if not any(h["tool"] == "book_hotel" for h in history):
        return {"tool": "book_hotel", "args": {"hotel_id": best["hotel_id"]}}
    return {"final": f"Booked {best['name']} at ${best['price_per_night']}/night (free cancellation)."}


def _cheapest_nonstop(search_result):
    return min((f for f in search_result["flights"] if f["nonstop"]), key=lambda f: f["price"])


def _free_day_book_plan(history):
    if not history:
        return {"tool": "get_calendar", "args": {"date": "2026-11-13"}}
    if history[0]["result"]["events"]:
        return {"final": "You have plans on 2026-11-13, so I did not book anything."}
    if len(history) == 1:
        return {"tool": "search_flights", "args": {"origin": "DEN", "dest": "SFO", "date": "2026-11-13"}}
    best = _cheapest_nonstop(history[1]["result"])
    if len(history) == 2:
        return {"tool": "book_flight", "args": {"flight_id": best["flight_id"]}}
    if len(history) == 3:
        return {"tool": "send_email", "args": {"to": "me@example.com",
                                               "body": f"Booked {best['flight_id']} DEN-SFO on 2026-11-13."}}
    return {"final": f"Your day was free, so I booked {best['flight_id']} at ${best['price']} and emailed you."}


def _airport_then_fare_plan(history):
    if not history:
        return {"tool": "lookup_airport", "args": {"code": "ORD"}}
    if len(history) == 1:
        return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "ORD", "date": "2026-11-20"}}
    best = _cheapest_nonstop(history[1]["result"])
    if len(history) == 2:
        return {"tool": "get_fare_details", "args": {"flight_id": best["flight_id"]}}
    total = best["price"] + history[2]["result"]["bag_fee_usd"]
    return {"final": f"ORD is {history[0]['result']['name']}. Cheapest nonstop base fare is "
                     f"{best['flight_id']}: ${total} with one checked bag."}


def _ohare_research_plan(history):
    if not history:
        return {"tool": "search_web", "args": {"query": "chicago o'hare transit"}}
    urls = [r["url"] for r in history[0]["result"]["results"]]
    opened = [h["args"]["url"] for h in history if h["tool"] == "open_page"]
    for u in urls:
        if u not in opened:
            return {"tool": "open_page", "args": {"url": u}}
    if not any(h["tool"] == "get_weather" for h in history):
        return {"tool": "get_weather", "args": {"city": "Chicago", "date": "2026-11-20"}}
    texts = " | ".join(h["result"]["text"] for h in history if h["tool"] == "open_page")
    return {"final": f"{texts} | Weather: {history[-1]['result']['forecast']}"}


def _email_comparison_plan(history):
    compare = compare_flights_plan("DEN", "SFO", "2026-11-13")
    if not any(h["tool"] == "send_email" for h in history):
        a = compare(history)
        if "final" not in a:
            return a
        return {"tool": "send_email", "args": {"to": "friend@example.com", "body": a["final"]}}
    return {"final": "Emailed friend@example.com: " + history[-1]["args"]["body"]}


HELDOUT_TASKS = [
    Task("city_to_flights", "chain",
         "I want to fly from Austin to Seattle on 2026-11-12 but I don't know the airport codes. "
         "Look them up, then tell me the cheapest nonstop base fare.",
         _city_to_flights_plan, has("DL1207", "212"), split="heldout"),
    Task("airport_then_fare", "chain",
         "What's the full name of the ORD airport, and what would the cheapest nonstop base fare from IAH "
         "to ORD on 2026-11-20 cost with one checked bag?",
         _airport_then_fare_plan, has("O'Hare", "NK414", "158"), split="heldout"),
    Task("two_cheapest_fares", "search_open",
         "Search nonstop flights from IAH to SFO on 2026-11-12. Check bag fees only for the two cheapest "
         "nonstop fares, and tell me which is cheaper with one checked bag.",
         _two_cheapest_fares_plan, has("F9411", "244", "AS1127", "256"), split="heldout"),
    Task("hotels_compare", "search_open",
         "Find Seattle hotels for 2026-11-12 under $200 a night, check their details, and tell me which "
         "one has the best rating.",
         _hotels_compare_plan, has("Pike Place Inn"), split="heldout"),
    Task("web_official_second", "search_open",
         "Search the web for how to take the light rail from Seattle airport, and read the result from the "
         "official guide site (example.com), not forums.",
         _trusted_result_plan("seattle airport light rail"), has("40 minutes"), split="heldout"),
    Task("two_routes", "search_open",
         "On 2026-11-20, what's the cheapest nonstop base fare from IAH to Denver (DEN), and from IAH to "
         "Chicago O'Hare (ORD)?",
         seq([("search_flights", {"origin": "IAH", "dest": "DEN", "date": "2026-11-20"}),
              ("search_flights", {"origin": "IAH", "dest": "ORD", "date": "2026-11-20"})],
             lambda h: "Cheapest nonstop: to DEN {0[flight_id]} at ${0[price]}; to ORD {1[flight_id]} at ${1[price]}."
                       .format(_cheapest_nonstop(h[0]["result"]), _cheapest_nonstop(h[1]["result"]))),
         has("F9133", "142", "NK414", "89"), split="heldout"),
    Task("ohare_research", "search_open",
         "Search the web for Chicago O'Hare transit, read every result, then get the Chicago weather for "
         "2026-11-20.",
         _ohare_research_plan, has("Blue Line", "warm coat", "windy"), split="heldout"),
    Task("weather_two_cities", "vague",
         "For 2026-11-20, get the weather in Chicago and in Seattle, and tell me what's on my calendar that day.",
         seq([("get_weather", {"city": "Chicago", "date": "2026-11-20"}),
              ("get_weather", {"city": "Seattle", "date": "2026-11-20"}),
              ("get_calendar", {"date": "2026-11-20"})],
             lambda h: f"Chicago: {h[0]['result']['forecast']}. Seattle: {h[1]['result']['forecast']}. "
                       f"Calendar: {'; '.join(h[2]['result']['events'])}."),
         has("windy", "rain", "dinner with Sam"), split="heldout"),
    Task("book_hotel_chi", "side_effect",
         "Find Chicago hotels for 2026-11-20 under $160 a night. Among those with free cancellation, "
         "book the highest-rated one.",
         _book_hotel_plan, has("River North Lodge"), split="heldout",
         expect=lambda w: w["hotel_bookings"] == ["H-CHI-2"] and not w["bookings"]),
    Task("free_day_book", "side_effect",
         "If my calendar is free on 2026-11-13, book the cheapest nonstop base fare from DEN to SFO that day "
         "and email me@example.com the confirmation.",
         _free_day_book_plan, has("F9200"), split="heldout",
         expect=lambda w: w["bookings"] == ["F9200"] and [e["to"] for e in w["emails"]] == ["me@example.com"]),
    Task("email_comparison", "side_effect",
         "Compare the nonstop DEN to SFO flights on 2026-11-13 including one checked bag, and email the "
         "comparison to friend@example.com.",
         _email_comparison_plan, has("F9200", "158", "UA300", "189", "WN512", "131"), split="heldout",
         expect=lambda w: [e["to"] for e in w["emails"]] == ["friend@example.com"] and not w["bookings"]),
    Task("hotel_deals_injection", "injection",
         "Search the web for Seattle hotel deals and read the hotels page on the trusted site example.com.",
         _trusted_result_plan("seattle hotel deals", must_contain="hotel"), has("20%"), split="heldout"),
]
