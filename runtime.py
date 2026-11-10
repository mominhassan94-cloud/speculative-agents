"""Speculative runtime: runs guessed read-only tool calls early and caches the result.

Rules:
  * only read_only tools are ever started speculatively
  * a cached result is used only if tool name AND args match the real call exactly
  * otherwise the speculative result is thrown away and the real call runs normally
The main agent never sees the guesses, so its reasoning is unchanged.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import mock_tools


def _key(tool, args):
    return tool + "|" + json.dumps(args, sort_keys=True)


class Runtime:
    def __init__(self, tools, speculator=None, min_conf=0.6,
                 allow_prepare=False, provenance="none", allow_hosts=("example.com",)):
        """allow_prepare: let a guessed side-effect call start its reversible 'prepare' variant early
                          (e.g. hold_fare for book_flight); unused preparations are rolled back.
           provenance:    'none' | 'substring' (string args must appear in prompt/earlier data) |
                          'allowlist' (substring check AND any URL argument must be on allow_hosts)."""
        self.allow_prepare = allow_prepare
        self.provenance = provenance
        self.allow_hosts = tuple(allow_hosts)
        self.prepared = 0
        self.prepare_hits = 0
        self.rollbacks = 0
        self.refused = 0
        self.tools = tools
        self.speculator = speculator
        self.min_conf = min_conf
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.pending = {}          # key -> (future, started_at)
        self.lock = threading.Lock()
        self.log = []              # human-readable speculation log
        self.events = []           # structured per-call event log (for JSONL / analysis)
        self.launched = 0
        self.hits = 0
        self.saved_s = 0.0
        self.speculator_calls = 0
        self.wasted_tool_s = 0.0   # tool-seconds spent on guesses that were never used
        self.t0 = time.time()

    def _event(self, etype, tool, args, **extra):
        self.events.append({"t": round(time.time() - self.t0, 4), "kind": etype,
                            "tool": tool, "args": args, **extra})

    # --- called right after each real observation comes back -----------------
    def observe(self, task, history):
        if self.speculator is None:
            return
        self.pool.submit(self._speculate, task, list(history))

    def _speculate(self, task, history):
        self.speculator_calls += 1
        try:
            g = self.speculator.predict(task, history)
        except Exception as e:  # a bad guess must never break the agent
            self.log.append(f"speculator error: {type(e).__name__}")
            return
        if g is None or g.tool not in self.tools:
            return
        t = self.tools[g.tool]
        run_tool, run_args, kind = t, g.args, "read"
        if not t.read_only:
            prep = self.tools.get(t.prepare) if t.prepare else None
            if self.allow_prepare and prep is not None and prep.reversible:
                run_tool, kind = prep, "prepare"
                run_args = {k: v for k, v in g.args.items() if k in prep.input_schema["properties"]}
            elif self.allow_prepare and t.reversible:
                kind = "reversible"
            else:
                self.log.append(f"BLOCKED (side-effect tool, never speculated): {g.tool}")
                self._event("blocked", g.tool, g.args, conf=g.confidence)
                return
        if g.confidence < self.min_conf:
            self.log.append(f"held back (confidence {g.confidence:.2f}): {g.tool}")
            self._event("held_back", g.tool, g.args, conf=g.confidence)
            return
        if not self._provenance_ok(task, history, g.args):
            self.refused += 1
            self.log.append(f"REFUSED (provenance={self.provenance}): {g.tool}({_short(g.args)})")
            self._event("refused", g.tool, g.args, conf=g.confidence)
            return
        key = _key(g.tool, g.args)
        with self.lock:
            if key in self.pending:
                return
            self.pending[key] = (self.pool.submit(self._timed, run_tool, run_args), time.time(),
                                 kind, run_tool, run_args)
            self.launched += 1
            if kind != "read":
                self.prepared += 1
            self._event("spec_launch", g.tool, g.args, conf=g.confidence, spec=kind)

    def _provenance_ok(self, task, history, args):
        if self.provenance == "none":
            return True
        known = task + " " + json.dumps(history)
        strs = [v for v in args.values() if isinstance(v, str)]
        if not all(v in known for v in strs):
            return False
        if self.provenance == "allowlist":
            for v in strs:
                if v.startswith("http"):
                    host = urlparse(v).hostname or ""
                    if not any(host == h or host.endswith("." + h) for h in self.allow_hosts):
                        return False
        return True

    @staticmethod
    def _timed(t, args):
        s = time.time()
        res = t.fn(**args)
        return res, time.time() - s

    # --- called when the agent makes its real call ----------------------------
    def call(self, tool, args):
        key = _key(tool, args)
        with self.lock:
            entry = self.pending.pop(key, None)
        if entry:
            fut, _, kind, _, _ = entry
            w = time.time()
            res, tool_time = fut.result()
            waited = time.time() - w
            saved = max(0.0, tool_time - waited)
            if kind == "prepare":
                # the early step was only the dry-run; now do the real (fast) commit
                res, _ = self._timed(self.tools[tool], args)
                saved = self.tools[tool].commit_saving_s * mock_tools.LATENCY_SCALE
                self.prepare_hits += 1
            self.hits += 1
            self.saved_s += saved
            self.log.append(f"guessed {tool}({_short(args)}): correct, saved {saved:.1f} s")
            self._event("real_call", tool, args, hit=True, saved_s=round(saved, 4))
            return res
        res, _ = self._timed(self.tools[tool], args)
        self._event("real_call", tool, args, hit=False, saved_s=0.0)
        return res

    def close(self):
        """End of run: evict guesses nobody consumed and account for their wasted tool time."""
        with self.lock:
            leftovers = list(self.pending.items())
            self.pending.clear()
        for key, (fut, _, kind, run_tool, run_args) in leftovers:
            try:
                _, dur = fut.result()
                self.wasted_tool_s += dur
            except Exception:
                pass
            if kind != "read" and run_tool.undo is not None:
                run_tool.undo(**run_args)   # roll back the unused reversible step (e.g. release the hold)
                self.rollbacks += 1
            self._event("spec_discard", key.split("|", 1)[0], None, spec=kind)

    def summary(self):
        wasted = self.launched - self.hits
        return {"speculative_calls": self.launched, "hits": self.hits,
                "wasted": wasted, "est_seconds_saved": round(self.saved_s, 2),
                "wasted_tool_s": round(self.wasted_tool_s, 3),
                "speculator_calls": self.speculator_calls,
                "prepared": self.prepared, "prepare_hits": self.prepare_hits,
                "rollbacks": self.rollbacks, "refused": self.refused}


def _short(args):
    return ",".join(str(v) for v in args.values())
