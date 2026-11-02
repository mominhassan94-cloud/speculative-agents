"""The agent loop plus two interchangeable 'brains':

  GeminiBrain    - real Gemini model via function calling (needs GEMINI_API_KEY)
  ScriptedBrain  - offline stand-in with a fixed plan and simulated thinking time,
                   used to test the pipeline and timing without an API key
"""
import json
import time


class ScriptedBrain:
    """Follows a fixed plan for the flight-comparison task. think_s mimics LLM latency."""
    name = "scripted"

    def __init__(self, think_s=1.5):
        self.think_s = think_s

    def next_action(self, task, history):
        time.sleep(self.think_s)
        if not history:
            return {"tool": "search_flights", "args": {"origin": "IAH", "dest": "DEN", "date": "2026-11-06"}}
        searches = [h for h in history if h["tool"] == "search_flights"]
        nonstop = [f["flight_id"] for f in searches[0]["result"]["flights"] if f["nonstop"]]
        done = [h["args"]["flight_id"] for h in history if h["tool"] == "get_fare_details"]
        for fid in nonstop:
            if fid not in done:
                return {"tool": "get_fare_details", "args": {"flight_id": fid}}
        fares = {h["args"]["flight_id"]: h["result"] for h in history if h["tool"] == "get_fare_details"}
        prices = {f["flight_id"]: f["price"] for f in searches[0]["result"]["flights"]}
        rows = [f"{fid}: ${prices[fid]} + ${fares[fid]['bag_fee_usd']} bag = ${prices[fid] + fares[fid]['bag_fee_usd']}"
                for fid in nonstop]
        return {"final": "Nonstop options with baggage: " + "; ".join(rows)}


class GeminiBrain:
    """Real Gemini model via function calling (needs GEMINI_API_KEY; model from GEMINI_MODEL)."""
    name = "gemini"

    def __init__(self, tools, model=None):
        from llm_client import LLMClient
        self.llm = LLMClient(model)
        self.model = self.llm.model
        self.config = self.llm.tool_config(tools)
        self.contents = []
        self.raw_log = []     # one entry per model call: what the model actually returned, plus timing

    def next_action(self, task, history):
        # Build the conversation incrementally so the model's own turns (with their thought
        # signatures) are sent back unchanged, instead of being re-serialized from `history`.
        if not history:
            self.contents = [self.llm.user_turn(task)]
        else:
            h = history[-1]
            self.contents.append(self.llm.tool_result_turn(h["tool"], h["result"]))
        t = time.time()
        content, action, raw = self.llm.step(self.contents, self.config)
        raw["llm_s"] = round(time.time() - t, 2)
        self.raw_log.append(raw)
        if content is not None:
            self.contents.append(content)
        return action


def run_agent(task, brain, runtime, max_steps=12, verbose=True):
    history = []
    t0 = time.time()
    for _ in range(max_steps):
        action = brain.next_action(task, history)   # speculation runs during this wait
        if "final" in action:
            break
        result = runtime.call(action["tool"], action["args"])
        history.append({"tool": action["tool"], "args": action["args"], "result": result})
        if verbose:
            print(f"  [{time.time() - t0:5.1f}s] {action['tool']}({json.dumps(action['args'])})")
        runtime.observe(task, history)              # start guessing the NEXT call immediately
    return action.get("final"), time.time() - t0, history
