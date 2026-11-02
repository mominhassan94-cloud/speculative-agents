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


def _key(tool, args):
    return tool + "|" + json.dumps(args, sort_keys=True)


class Runtime:
    def __init__(self, tools, speculator=None, min_conf=0.6):
        self.tools = tools
        self.speculator = speculator
        self.min_conf = min_conf
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.pending = {}          # key -> (future, started_at)
        self.lock = threading.Lock()
        self.log = []              # human-readable speculation log
        self.launched = 0
        self.hits = 0
        self.misses = 0
        self.too_late = 0
        self.saved_s = 0.0
        self.step = 0              # number of real tool calls made so far
        self.t0 = time.time()

    # --- called right after each real observation comes back -----------------
    def observe(self, task, history):
        if self.speculator is None:
            return
        self.pool.submit(self._speculate, task, list(history), self.step)

    def _note(self, msg):
        self.log.append(f"[{time.time() - self.t0:5.1f}s] {msg}")

    def _speculate(self, task, history, step):
        try:
            g = self.speculator.predict(task, history)
        except Exception as e:  # a bad guess must never break the agent
            self._note(f"speculator error: {type(e).__name__}: {e}")
            return
        if g is None:
            self._note("no guess (speculator returned null)")
            return
        if g.tool not in self.tools:
            self._note(f"no guess (unknown tool {g.tool!r})")
            return
        t = self.tools[g.tool]
        if not t.read_only:
            self._note(f"BLOCKED (side-effect tool, never speculated): {g.tool}")
            return
        if g.confidence < self.min_conf:
            self._note(f"held back (confidence {g.confidence:.2f}): {g.tool}({_short(g.args)})")
            return
        key = _key(g.tool, g.args)
        with self.lock:
            if self.step != step:  # the agent already made its next call; this guess is stale
                self.too_late += 1
                self._note(f"too late (agent already moved on): {g.tool}({_short(g.args)})")
                return
            if key in self.pending:
                return
            self.pending[key] = (self.pool.submit(self._timed, t, g.args), time.time())
            self.launched += 1
            self._note(f"launched guess {g.tool}({_short(g.args)}) conf {g.confidence:.2f}")

    @staticmethod
    def _timed(t, args):
        s = time.time()
        res = t.fn(**args)
        return res, time.time() - s

    # --- called when the agent makes its real call ----------------------------
    def call(self, tool, args):
        key = _key(tool, args)
        with self.lock:
            self.step += 1
            entry = self.pending.pop(key, None)
            stale = list(self.pending)
            self.pending.clear()
        for k in stale:  # guesses that did not match this real call are discarded
            self.misses += 1
            self._note(f"guessed {k.replace('|', ' ')}: WRONG (real call {tool}({_short(args)})), discarded")
        if entry:
            fut, _ = entry
            w = time.time()
            res, tool_time = fut.result()
            waited = time.time() - w
            saved = max(0.0, tool_time - waited)
            self.hits += 1
            self.saved_s += saved
            self._note(f"guessed {tool}({_short(args)}): correct, saved {saved:.1f} s")
            return res
        res, _ = self._timed(self.tools[tool], args)
        return res

    def summary(self):
        wasted = self.launched - self.hits
        return {"speculative_calls": self.launched, "hits": self.hits, "misses": self.misses,
                "wasted": wasted, "too_late": self.too_late, "est_seconds_saved": round(self.saved_s, 2)}


def _short(args):
    return ",".join(str(v) for v in args.values())
