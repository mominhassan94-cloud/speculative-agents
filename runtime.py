"""Speculative runtime: runs guessed read-only tool calls early and caches the result.

Rules:
  * only read_only tools are ever started speculatively
  * a cached result is used only if tool name AND args match the real call exactly
  * otherwise the speculative result is thrown away and the real call runs normally
The main agent never sees the guesses, so its reasoning is unchanged.
"""
import json
import threading
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import mock_tools


def _key(tool, args):
    return tool + "|" + json.dumps(args, sort_keys=True)


class Runtime:
    def __init__(self, tools, speculator=None, min_conf=0.6,
                 allow_prepare=False, provenance="none", allow_hosts=("example.com",), cache="step"):
        """allow_prepare: let a guessed side-effect call start its reversible 'prepare' variant early
                          (e.g. hold_fare for book_flight); unused preparations are rolled back.
           provenance:    'none' | 'substring' (string args must appear in prompt/earlier data) |
                          'allowlist' (substring check AND any URL argument must be on allow_hosts).
           cache:         how long an unused guess may wait for a matching real call (H8, stale reads):
                          'step'       until the next real call (default; can never serve a read across a write)
                          'run'        until the end of the run (the pre-merge behaviour; can serve stale reads)
                          'invalidate' until the end of the run, but a write drops cached reads it may affect
                                       (same scope, or either scope unknown)."""
        assert cache in ("step", "run", "invalidate"), cache
        self.cache = cache
        self.writes = defaultdict(int)   # scope -> real writes so far ("*" counts every write)
        self.launch_writes = {}          # key -> snapshot of self.writes when the guess was launched
        self.stale_served = 0            # hits whose read was launched before a write that may affect it
        self.invalidated = 0
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
        self.pending = {}          # key -> (future, started_at, kind, run_tool, run_args)
        self.discarded = []        # (key, entry) for guesses dropped on a miss; settled in close()
        self.lock = threading.Lock()
        self.log = []              # human-readable speculation log
        self.events = []           # structured per-call event log (for JSONL / analysis)
        self.launched = 0
        self.hits = 0
        self.misses = 0
        self.too_late = 0
        self.saved_s = 0.0
        self.step = 0              # number of real tool calls made so far
        self.speculator_calls = 0
        self.wasted_tool_s = 0.0   # tool-seconds spent on guesses that were never used
        self.t0 = time.perf_counter()

    def _now(self):
        return time.perf_counter() - self.t0

    def _note(self, msg):
        self.log.append(f"[{self._now():5.1f}s] {msg}")

    def _event(self, etype, tool, args, **extra):
        self.events.append({"t": round(self._now(), 4), "kind": etype,
                            "tool": tool, "args": args, **extra})

    # --- called right after each real observation comes back -----------------
    def observe(self, task, history):
        if self.speculator is None:
            return
        self.pool.submit(self._speculate, task, list(history), self.step)

    def _speculate(self, task, history, step):
        self.speculator_calls += 1
        try:
            g = self.speculator.predict(task, history)
        except Exception as e:  # a bad guess must never break the agent
            self._note(f"speculator error: {type(e).__name__}: {e}")
            return
        if g is None:
            self._note("no guess (speculator returned null)")
            self._event("no_guess", None, None, step=step)
            return
        # every guess is logged before any gate, so accuracy can be scored against the next real call
        self._event("guess", g.tool, g.args, conf=g.confidence, step=step)
        if g.tool not in self.tools:
            self._note(f"no guess (unknown tool {g.tool!r})")
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
                self._note(f"BLOCKED (side-effect tool, never speculated): {g.tool}")
                self._event("blocked", g.tool, g.args, conf=g.confidence)
                return
        if g.confidence < self.min_conf:
            self._note(f"held back (confidence {g.confidence:.2f}): {g.tool}({_short(g.args)})")
            self._event("held_back", g.tool, g.args, conf=g.confidence)
            return
        if not self._provenance_ok(task, history, g.args):
            self.refused += 1
            self._note(f"REFUSED (provenance={self.provenance}): {g.tool}({_short(g.args)})")
            self._event("refused", g.tool, g.args, conf=g.confidence)
            return
        key = _key(g.tool, g.args)
        with self.lock:
            if self.step != step:  # the agent already made its next call; this guess is stale
                self.too_late += 1
                self._note(f"too late (agent already moved on): {g.tool}({_short(g.args)})")
                self._event("too_late", g.tool, g.args, conf=g.confidence)
                return
            if key in self.pending:
                return
            self.pending[key] = (self.pool.submit(self._timed, run_tool, run_args), time.perf_counter(),
                                 kind, run_tool, run_args)
            self.launch_writes[key] = dict(self.writes)
            self.launched += 1
            if kind != "read":
                self.prepared += 1
            self._note(f"launched guess {g.tool}({_short(g.args)}) conf {g.confidence:.2f} [{kind}]")
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
        s = time.perf_counter()
        res = t.fn(**args)
        return res, time.perf_counter() - s

    # --- called when the agent makes its real call ----------------------------
    def call(self, tool, args):
        key = _key(tool, args)
        t = self.tools[tool]
        with self.lock:
            self.step += 1
            entry = self.pending.pop(key, None)
            launched_at = self.launch_writes.pop(key, {})
            if self.cache == "step":
                stale = list(self.pending.items())
            elif self.cache == "invalidate" and not t.read_only:
                stale = [(k, e) for k, e in self.pending.items() if _may_affect(t.scope, e[3].scope)]
                self.invalidated += len(stale)
            else:
                stale = []
            for k, _ in stale:
                del self.pending[k]
                self.launch_writes.pop(k, None)
            self.discarded.extend(stale)   # wasted time and rollback are settled in close()
            if not t.read_only:
                self.writes[t.scope or "?"] += 1
                self.writes["*"] += 1
        if entry and entry[2] == "read" and self._written_since(launched_at, t.scope):
            self.stale_served += 1     # this cached read predates a write that may have changed it
            self._note(f"STALE read served: {tool}({_short(args)})")
        for k, _ in stale:  # guesses that did not match this real call are discarded
            self.misses += 1
            self._note(f"guessed {k.replace('|', ' ')}: WRONG (real call {tool}({_short(args)})), discarded")
        if entry:
            fut, _, kind, _, _ = entry
            w = time.perf_counter()
            res, tool_time = fut.result()
            waited = time.perf_counter() - w
            saved = max(0.0, tool_time - waited)
            if kind == "prepare":
                # the early step was only the dry-run; now do the real (fast) commit
                res, _ = self._timed(self.tools[tool], args)
                saved = self.tools[tool].commit_saving_s * mock_tools.LATENCY_SCALE
                self.prepare_hits += 1
            self.hits += 1
            self.saved_s += saved
            self._note(f"guessed {tool}({_short(args)}): correct, saved {saved:.1f} s")
            self._event("real_call", tool, args, hit=True, saved_s=round(saved, 4), step=self.step)
            return res
        res, _ = self._timed(self.tools[tool], args)
        self._event("real_call", tool, args, hit=False, saved_s=0.0, step=self.step)
        return res

    def _written_since(self, snapshot, scope):
        """Did a write that may affect a read of this scope happen after the snapshot was taken?"""
        if scope is None:
            return self.writes["*"] != snapshot.get("*", 0)
        return any(self.writes[s] != snapshot.get(s, 0) for s in (scope, "?"))

    def close(self):
        """End of run: evict guesses nobody consumed and account for their wasted tool time."""
        with self.lock:
            leftovers = self.discarded + list(self.pending.items())
            self.discarded = []
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
        return {"speculative_calls": self.launched, "hits": self.hits, "misses": self.misses,
                "wasted": wasted, "too_late": self.too_late, "est_seconds_saved": round(self.saved_s, 2),
                "wasted_tool_s": round(self.wasted_tool_s, 3),
                "speculator_calls": self.speculator_calls,
                "prepared": self.prepared, "prepare_hits": self.prepare_hits,
                "rollbacks": self.rollbacks, "refused": self.refused,
                "stale_served": self.stale_served, "invalidated": self.invalidated}


def _may_affect(write_scope, read_scope):
    """Could a write in write_scope change a read in read_scope? Unknown scopes are assumed to overlap."""
    return write_scope is None or read_scope is None or write_scope == read_scope


def _short(args):
    return ",".join(str(v) for v in args.values())
