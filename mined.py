"""Track C: a speculator mined from traces instead of written by hand (H3, H4).

Fit on call traces from TRAIN tasks only, then guess on anything:
  next tool:  counts of what followed the last two tools (backing off to the last tool), incl. 'stop'
  arguments:  each argument gets the template that explained it most often in training:
      list   the first item of a list in an earlier result whose field was not used yet, keeping only items
             that match boolean filters the training choices always had (e.g. nonstop=True)
      field  a scalar field of an earlier result
      arg    an argument of an earlier call (e.g. reuse the date)
      const  a value that was the same in every training example
Confidence = P(next tool | context) x the argument templates' training reliability.

    python mined.py          # fit on train traces and print what was learned
"""
import json
from collections import Counter, defaultdict

import mock_tools
from mock_tools import TOOLS
from speculator import Guess

STOP = "<stop>"


def collect_traces(tasks):
    """Run each task's plan against the real (mock) tools with no latency; return [(prompt, history)]."""
    old = mock_tools.LATENCY_SCALE
    mock_tools.LATENCY_SCALE = 0.0
    traces = []
    try:
        for t in tasks:
            mock_tools.reset_world()
            history = []
            for _ in range(12):
                a = t.plan(history)
                if "final" in a:
                    break
                history.append({"tool": a["tool"], "args": a["args"], "result": TOOLS[a["tool"]].fn(**a["args"])})
            traces.append((t.prompt, history))
    finally:
        mock_tools.LATENCY_SCALE = old
        mock_tools.reset_world()
    return traces


def _contexts(history):
    """Lookup keys from most to least specific."""
    tools = [h["tool"] for h in history]
    return [tuple(tools[-2:]), tuple(tools[-1:])]


def _explain(history, name, value):
    """All templates that would have produced value for argument `name` given this history."""
    out = []
    for h in reversed(history):
        res = h["result"] if isinstance(h["result"], dict) else {}
        for key, items in res.items():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        for field, v in it.items():
                            if v == value:
                                out.append((("list", h["tool"], key, field), it, items))
            elif items == value:
                out.append((("field", h["tool"], key), None, None))
        for a, v in h["args"].items():
            if v == value:
                out.append((("arg", h["tool"], a), None, None))
    out.append((("const", json.dumps(value)), None, None))
    return out


class MinedSpeculator:
    name = "mined"

    def __init__(self, min_count=2):
        self.min_count = min_count
        self.next_tool = defaultdict(Counter)       # context -> Counter(next tool)
        self.templates = defaultdict(Counter)       # (tool, arg) -> Counter(template)
        self.seen = Counter()                       # (tool, arg) -> examples
        self.chosen_flags = defaultdict(list)       # list template -> [bool fields of each chosen item]

    # --- learning -----------------------------------------------------------
    def fit(self, traces):
        for _, history in traces:
            for i in range(1, len(history) + 1):
                past = history[:i]
                nxt = history[i] if i < len(history) else None
                for ctx in _contexts(past):
                    self.next_tool[ctx][nxt["tool"] if nxt else STOP] += 1
                if nxt is None:
                    continue
                for name, value in nxt["args"].items():
                    self.seen[(nxt["tool"], name)] += 1
                    for tmpl, item, _ in {e[0]: e for e in _explain(past, name, value)}.values():
                        self.templates[(nxt["tool"], name)][tmpl] += 1
                        if item is not None:
                            self.chosen_flags[tmpl].append({k: v for k, v in item.items() if isinstance(v, bool)})
        return self

    def _best_template(self, tool, name):
        c = self.templates.get((tool, name))
        if not c:
            return None, 0.0
        # prefer data-derived templates over constants when they explain as many examples
        tmpl, n = max(c.items(), key=lambda kv: (kv[1], kv[0][0] != "const"))
        return tmpl, n / self.seen[(tool, name)]

    def _filters(self, tmpl):
        flags = self.chosen_flags.get(tmpl, [])
        if not flags:
            return {}
        return {k: v for k, v in flags[0].items() if all(f.get(k) == v for f in flags)}

    # --- guessing -----------------------------------------------------------
    def predict(self, task, history):
        if not history:
            return None
        counts = next((self.next_tool[c] for c in _contexts(history)
                       if sum(self.next_tool[c].values()) >= self.min_count), None)
        if not counts:
            return None
        tool, n = counts.most_common(1)[0]
        if tool == STOP or tool not in TOOLS:
            return None
        conf = n / sum(counts.values())
        args = {}
        for name in TOOLS[tool].input_schema["properties"]:
            tmpl, rel = self._best_template(tool, name)
            v = self._apply(tmpl, tool, name, history) if tmpl else None
            if v is None:
                return None
            args[name] = v
            conf *= rel
        return Guess(tool, args, round(conf, 3))

    def _apply(self, tmpl, tool, name, history):
        kind = tmpl[0]
        if kind == "const":
            return json.loads(tmpl[1])
        src_i = next((i for i in range(len(history) - 1, -1, -1) if history[i]["tool"] == tmpl[1]), None)
        if src_i is None:
            return None
        src = history[src_i]
        if kind == "arg":
            return src["args"].get(tmpl[2])
        res = src["result"] if isinstance(src["result"], dict) else {}
        if kind == "field":
            return res.get(tmpl[2])
        used = {h["args"].get(name) for h in history[src_i + 1:] if h["tool"] == tool}
        flt = self._filters(tmpl)
        for it in res.get(tmpl[2], []):
            if isinstance(it, dict) and all(it.get(k) == v for k, v in flt.items()) and it.get(tmpl[3]) not in used:
                return it.get(tmpl[3])
        return None


def fit_on_train():
    from tasks import TRAIN_TASKS
    return MinedSpeculator().fit(collect_traces(TRAIN_TASKS))


if __name__ == "__main__":
    m = fit_on_train()
    print("next-tool table (last tool -> next):")
    for ctx, c in sorted(m.next_tool.items(), key=lambda kv: str(kv[0])):
        if len(ctx) == 1:
            print(f"  {ctx[0]:<18} -> {dict(c.most_common())}")
    print("\nargument templates:")
    for (tool, name) in sorted(m.seen):
        tmpl, rel = m._best_template(tool, name)
        flt = m._filters(tmpl) if tmpl and tmpl[0] == "list" else {}
        print(f"  {tool}.{name:<10} {str(tmpl):<55} rel {rel:.2f} {flt or ''}")
