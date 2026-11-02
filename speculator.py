"""Speculators: guess the agent's next tool call from the history so far.

Two versions, so the project can compare a cheap LLM against simple rules:
  - RuleSpeculator: hand-written rules (free, instant)
  - LLMSpeculator:  a small, cheap model (costs money, adds a little latency)
"""
import json
import re
from dataclasses import dataclass


@dataclass
class Guess:
    tool: str
    args: dict
    confidence: float


class RuleSpeculator:
    name = "rules"

    def predict(self, task, history):
        if not history:
            return None
        last = history[-1]
        if last["tool"] == "search_flights":
            flights = last["result"].get("flights", [])
            nonstop = [f for f in flights if f.get("nonstop")]
            if nonstop:
                return Guess("get_fare_details", {"flight_id": nonstop[0]["flight_id"]}, 0.8)
        if last["tool"] == "get_fare_details":
            # next unvisited nonstop flight from the most recent search
            search = next((h for h in reversed(history) if h["tool"] == "search_flights"), None)
            if search:
                seen = {h["args"].get("flight_id") for h in history if h["tool"] == "get_fare_details"}
                for f in search["result"].get("flights", []):
                    if f.get("nonstop") and f["flight_id"] not in seen:
                        return Guess("get_fare_details", {"flight_id": f["flight_id"]}, 0.7)
        if last["tool"] == "search_web":
            results = last["result"].get("results", [])
            if results:
                return Guess("open_page", {"url": results[0]["url"]}, 0.75)
        return None


SPEC_PROMPT = """You predict the NEXT tool call an AI agent will make, so it can be started early.

You are given the user's task, the tools available, and the calls made so far with their results.
Predict the single most likely next tool call, with exact arguments copied from the data above
(do not invent values). If the agent is probably finished, or you cannot predict exact arguments,
return null for tool.

Reply with ONLY JSON, no other text:
{"tool": "<tool name or null>", "args": {...}, "confidence": <0.0-1.0>}"""


class LLMSpeculator:
    name = "llm"

    def __init__(self, tools, model="claude-haiku-5-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.tool_desc = "\n".join(f"- {t.name}({', '.join(t.input_schema['properties'])}): {t.description}"
                                   for t in tools.values())
        self.last_raw = None

    def _complete(self, user):
        r = self.client.messages.create(model=self.model, max_tokens=200, system=SPEC_PROMPT,
                                        messages=[{"role": "user", "content": user}])
        return r.content[0].text

    def predict(self, task, history):
        if not history:
            return None
        hist = "\n".join(f"{i+1}. {h['tool']}({json.dumps(h['args'])}) -> {json.dumps(h['result'])}"
                         for i, h in enumerate(history))
        user = f"TASK: {task}\n\nTOOLS:\n{self.tool_desc}\n\nCALLS SO FAR:\n{hist}\n\nPredict the next call."
        self.last_user = user
        raw = self._complete(user)
        self.last_raw = raw
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not d.get("tool"):
            return None
        return Guess(d["tool"], d.get("args", {}), float(d.get("confidence", 0)))


class GeminiSpeculator(LLMSpeculator):
    """Same guesser, but using a Gemini model. Needs GEMINI_API_KEY (or GOOGLE_API_KEY) set."""
    name = "gemini"

    def __init__(self, tools, model=None):
        import os
        from google import genai
        self.genai = genai
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.tool_desc = "\n".join(f"- {t.name}({', '.join(t.input_schema['properties'])}): {t.description}"
                                   for t in tools.values())
        self.last_raw = None

    def _complete(self, user):
        from google.genai import types
        r = self.client.models.generate_content(
            model=self.model, contents=user,
            config=types.GenerateContentConfig(system_instruction=SPEC_PROMPT, max_output_tokens=300))
        return r.text
