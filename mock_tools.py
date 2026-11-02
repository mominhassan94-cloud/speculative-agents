"""Mock tools: plain Python functions with simulated API latency.

Each tool is registered with a read_only flag. Only read_only tools may ever be
run speculatively. Side-effecting tools (booking, sending) never run early.
"""
import time
from dataclasses import dataclass
from typing import Callable

LATENCY_SCALE = 1.0  # set <1.0 to speed up experiments


@dataclass
class Tool:
    name: str
    fn: Callable
    read_only: bool
    latency_s: float
    description: str
    input_schema: dict


TOOLS: dict[str, Tool] = {}


def tool(name, read_only, latency_s, description, props, required=None):
    def deco(fn):
        def wrapped(**kwargs):
            time.sleep(latency_s * LATENCY_SCALE)  # pretend this is a network call
            return fn(**kwargs)
        TOOLS[name] = Tool(
            name, wrapped, read_only, latency_s, description,
            {"type": "object", "properties": props, "required": required or list(props)},
        )
        return wrapped
    return deco


_S = {"type": "string"}

FLIGHTS = {
    ("IAH", "DEN"): [
        {"flight_id": "UA482", "airline": "United", "price": 189, "nonstop": True},
        {"flight_id": "WN219", "airline": "Southwest", "price": 204, "nonstop": True},
        {"flight_id": "F9133", "airline": "Frontier", "price": 142, "nonstop": True},
        {"flight_id": "AA907", "airline": "American", "price": 231, "nonstop": False},
    ],
}
FARE_DETAILS = {
    "UA482": {"flight_id": "UA482", "bag_fee_usd": 40, "carry_on_fee_usd": 0},
    "WN219": {"flight_id": "WN219", "bag_fee_usd": 0, "carry_on_fee_usd": 0},
    "F9133": {"flight_id": "F9133", "bag_fee_usd": 65, "carry_on_fee_usd": 55},
    "AA907": {"flight_id": "AA907", "bag_fee_usd": 40, "carry_on_fee_usd": 0},
}
PAGES = {
    "https://example.com/den-guide": "Denver airport guide: ground transport, lounges, parking.",
    "https://example.com/bag-rules": "Checked bag fees vary by fare class and airline.",
}


@tool("search_flights", True, 2.0, "Search nonstop and connecting flights between two airports on a date.",
      {"origin": _S, "dest": _S, "date": _S})
def search_flights(origin, dest, date):
    return {"flights": FLIGHTS.get((origin, dest), [])}


@tool("get_fare_details", True, 1.5, "Get baggage fees and fare rules for one flight.",
      {"flight_id": _S})
def get_fare_details(flight_id):
    return FARE_DETAILS.get(flight_id, {"error": "unknown flight"})


@tool("search_web", True, 1.8, "Search the web. Returns a list of result links.", {"query": _S})
def search_web(query):
    return {"results": [{"url": u, "snippet": s[:40]} for u, s in PAGES.items()]}


@tool("open_page", True, 1.2, "Open a web page by URL and return its text.", {"url": _S})
def open_page(url):
    return {"url": url, "text": PAGES.get(url, "404")}


@tool("get_weather", True, 0.9, "Get the weather forecast for a city and date.", {"city": _S, "date": _S})
def get_weather(city, date):
    return {"city": city, "date": date, "forecast": "sunny, 18C"}


@tool("lookup_airport", True, 0.7, "Look up an airport by IATA code.", {"code": _S})
def lookup_airport(code):
    return {"code": code, "name": {"IAH": "Houston Intercontinental", "DEN": "Denver International"}.get(code, "?")}


@tool("get_calendar", True, 1.0, "Read the user's calendar for a date.", {"date": _S})
def get_calendar(date):
    return {"date": date, "events": ["10:00 team meeting"]}


@tool("book_flight", False, 2.0, "BOOK a flight (charges the user). Side effect.", {"flight_id": _S})
def book_flight(flight_id):
    return {"status": "booked", "flight_id": flight_id}


@tool("send_email", False, 1.5, "SEND an email. Side effect.", {"to": _S, "body": _S})
def send_email(to, body):
    return {"status": "sent", "to": to}
