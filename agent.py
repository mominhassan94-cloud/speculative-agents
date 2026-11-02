"""The agent loop plus two interchangeable 'brains':

  AnthropicBrain - real Claude via tool use (needs ANTHROPIC_API_KEY)
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


class AnthropicBrain:
    name = "claude"

    def __init__(self, tools, model="claude-sonnet-5-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                          for t in tools.values()]

    def next_action(self, task, history):
        msgs = [{"role": "user", "content": task}]
        for i, h in enumerate(history):
            tid = f"toolu_{i}"
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": h["tool"], "input": h["args"]}]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": json.dumps(h["result"])}]})
        r = self.client.messages.create(model=self.model, max_tokens=1024, tools=self.tool_defs, messages=msgs)
        for b in r.content:
            if b.type == "tool_use":
                return {"tool": b.name, "args": b.input}
        return {"final": "".join(b.text for b in r.content if b.type == "text")}


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
